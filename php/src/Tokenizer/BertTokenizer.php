<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tokenizer;

use InvalidArgumentException;
use JsonException;
use Normalizer;
use Omegaalfa\Transformer\Tensor\Shape;
use Omegaalfa\Transformer\Transformer\AttentionMask;
use RuntimeException;

final readonly class BertTokenizer implements TokenizerInterface
{
    /** @var array<string, int> */
    private array $vocabulary;

    /** @var array<int, string> */
    private array $tokens;

    private int $unknownId;
    private int $classificationId;
    private int $separatorId;
    private int $paddingId;

    /** @param array<string, int> $vocabulary */
    private function __construct(array $vocabulary, public int $maximumLength = 512)
    {
        if ($maximumLength < 2) {
            throw new InvalidArgumentException('BERT tokenizer maximum length must allow CLS and SEP.');
        }
        foreach ($vocabulary as $token => $id) {
            if ($token === '' || $id < 0) {
                throw new InvalidArgumentException('BERT vocabulary entries must have non-empty tokens and non-negative IDs.');
            }
        }
        $this->vocabulary = $vocabulary;
        $this->tokens = array_flip($vocabulary);
        $this->unknownId = $this->requiredId('[UNK]');
        $this->classificationId = $this->requiredId('[CLS]');
        $this->separatorId = $this->requiredId('[SEP]');
        $this->paddingId = $this->requiredId('[PAD]');
    }

    public static function fromTokenizerJson(string $path, int $maximumLength = 512): self
    {
        $json = @file_get_contents($path);
        if ($json === false) {
            throw new RuntimeException("Unable to read BERT tokenizer: {$path}");
        }
        try {
            $data = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
        } catch (JsonException $exception) {
            throw new RuntimeException('Invalid tokenizer JSON.', previous: $exception);
        }
        $model = is_array($data) ? ($data['model'] ?? null) : null;
        $vocabulary = is_array($model) ? ($model['vocab'] ?? null) : null;
        if (!is_array($model) || ($model['type'] ?? null) !== 'WordPiece'
            || ($model['unk_token'] ?? null) !== '[UNK]'
            || ($model['continuing_subword_prefix'] ?? null) !== '##'
            || !is_array($vocabulary) || array_is_list($vocabulary)) {
            throw new RuntimeException('Tokenizer must contain the approved BERT WordPiece model.');
        }
        $validated = [];
        foreach ($vocabulary as $token => $id) {
            if (!is_int($id)) {
                throw new RuntimeException('Tokenizer vocabulary must map strings to integer IDs.');
            }
            $validated[(string) $token] = $id;
        }

        return new self($validated, $maximumLength);
    }

    public function encode(string $text): TokenizationResult
    {
        $ids = [$this->classificationId];
        foreach ($this->basicTokens($text) as $token) {
            foreach ($this->wordPieces($token) as $id) {
                if (count($ids) >= $this->maximumLength - 1) {
                    break 2;
                }
                $ids[] = $id;
            }
        }
        $ids[] = $this->separatorId;

        return new TokenizationResult($ids, array_fill(0, count($ids), 1), array_fill(0, count($ids), 0));
    }

    /** @param array<array-key, mixed> $texts */
    public function encodeBatch(array $texts): BertBatchEncoding
    {
        if ($texts === [] || !array_is_list($texts)) {
            throw new InvalidArgumentException('BERT tokenizer batch must be a non-empty list of strings.');
        }
        $encoded = [];
        $sequence = 0;
        foreach ($texts as $text) {
            if (!is_string($text)) {
                throw new InvalidArgumentException('BERT tokenizer inputs must be strings.');
            }
            $result = $this->encode($text);
            $encoded[] = $result;
            $sequence = max($sequence, count($result->tokenIds));
        }
        $inputIds = [];
        $mask = [];
        $tokenTypes = [];
        foreach ($encoded as $result) {
            $length = count($result->tokenIds);
            $padding = $sequence - $length;
            array_push($inputIds, ...$result->tokenIds, ...array_fill(0, $padding, $this->paddingId));
            array_push($mask, ...array_fill(0, $length, true), ...array_fill(0, $padding, false));
            array_push($tokenTypes, ...array_fill(0, $sequence, 0));
        }
        $shape = new Shape([count($encoded), $sequence]);

        return new BertBatchEncoding($inputIds, new AttentionMask($mask, $shape), $tokenTypes, $shape);
    }

    /** @param array<array-key, mixed> $tokenIds */
    public function decode(array $tokenIds): string
    {
        if (!array_is_list($tokenIds)) {
            throw new InvalidArgumentException('Token IDs must be a list.');
        }
        $text = '';
        foreach ($tokenIds as $id) {
            if (!is_int($id) || !isset($this->tokens[$id])) {
                throw new InvalidArgumentException('Token ID is outside the BERT vocabulary.');
            }
            $token = $this->tokens[$id];
            if (in_array($token, ['[CLS]', '[SEP]', '[PAD]'], true)) {
                continue;
            }
            if (str_starts_with($token, '##')) {
                $text .= substr($token, 2);
            } else {
                $text .= ($text === '' ? '' : ' ') . $token;
            }
        }
        return $text;
    }

    /** @return list<string> */
    private function basicTokens(string $text): array
    {
        if (!mb_check_encoding($text, 'UTF-8')) {
            throw new InvalidArgumentException('Tokenizer input must be valid UTF-8.');
        }
        $text = preg_replace('/[\p{Cc}\p{Cf}]/u', '', $text);
        if (!is_string($text)) {
            throw new InvalidArgumentException('Unable to clean tokenizer input.');
        }
        $text = $this->spaceChineseCharacters($text);
        $normalized = Normalizer::normalize(mb_strtolower($text, 'UTF-8'), Normalizer::FORM_D);
        if (!is_string($normalized)) {
            throw new InvalidArgumentException('Unable to normalize tokenizer input.');
        }
        $normalized = preg_replace('/\p{Mn}+/u', '', $normalized);
        if (!is_string($normalized)) {
            throw new InvalidArgumentException('Unable to strip tokenizer accents.');
        }
        $pieces = preg_split('/\s+/u', trim($normalized), -1, PREG_SPLIT_NO_EMPTY);
        if ($pieces === false) {
            throw new InvalidArgumentException('Unable to split tokenizer input.');
        }
        $tokens = [];
        foreach ($pieces as $piece) {
            $split = preg_split('/([\p{P}\p{S}])/u', $piece, -1, PREG_SPLIT_NO_EMPTY | PREG_SPLIT_DELIM_CAPTURE);
            if ($split === false) {
                throw new InvalidArgumentException('Unable to split tokenizer punctuation.');
            }
            array_push($tokens, ...$split);
        }
        return $tokens;
    }

    /** @return list<int> */
    private function wordPieces(string $token): array
    {
        $characters = preg_split('//u', $token, -1, PREG_SPLIT_NO_EMPTY);
        if ($characters === false || count($characters) > 100) {
            return [$this->unknownId];
        }
        $ids = [];
        $start = 0;
        $length = count($characters);
        while ($start < $length) {
            $matched = null;
            $end = $length;
            while ($end > $start) {
                $candidate = ($start === 0 ? '' : '##') . implode('', array_slice($characters, $start, $end - $start));
                if (isset($this->vocabulary[$candidate])) {
                    $matched = $this->vocabulary[$candidate];
                    break;
                }
                --$end;
            }
            if ($matched === null) {
                return [$this->unknownId];
            }
            $ids[] = $matched;
            $start = $end;
        }
        return $ids;
    }

    private function spaceChineseCharacters(string $text): string
    {
        $spaced = preg_replace('/([\x{3400}-\x{4DBF}\x{4E00}-\x{9FFF}\x{F900}-\x{FAFF}])/u', ' $1 ', $text);
        if ($spaced === null) {
            throw new InvalidArgumentException('Unable to tokenize Chinese characters.');
        }
        return $spaced;
    }

    private function requiredId(string $token): int
    {
        return $this->vocabulary[$token] ?? throw new InvalidArgumentException("BERT vocabulary is missing {$token}.");
    }
}
