<?php

declare(strict_types=1);

namespace Omegaalfa\Transformer\Tests\Unit\Model;

use Omegaalfa\Transformer\Exception\ModelException;
use Omegaalfa\Transformer\Model\Encoder\Bert\BertConfigReader;
use Omegaalfa\Transformer\Model\Loader\BgeSmallEnV15Manifest;
use Omegaalfa\Transformer\Model\Loader\WeightOrientation;
use PHPUnit\Framework\TestCase;

final class BertConfigAndManifestTest extends TestCase
{
    public function testReadsBgeConfigAndBuildsCompleteClosedManifest(): void
    {
        $path = $this->configFile();
        try {
            $config = (new BertConfigReader())->read($path);
            self::assertSame(30522, $config->vocabularySize);
            self::assertSame(384, $config->hiddenSize);
            self::assertSame(12, $config->numHiddenLayers);
            $manifest = BgeSmallEnV15Manifest::create($config);
            self::assertCount(197, $manifest->parameters);
            self::assertSame(['embeddings.position_ids', 'pooler.dense.weight', 'pooler.dense.bias'], $manifest->ignoredCheckpointTensors);
            self::assertSame(WeightOrientation::PyTorchLinearTranspose, $manifest->parameters[5]->materialization->orientation);
        } finally {
            @unlink($path);
        }
    }

    public function testRejectsMissingAndUnsupportedConfigFields(): void
    {
        foreach ([['hidden_size' => 384], $this->config(['hidden_act' => 'gelu_new'])] as $data) {
            $path = tempnam(sys_get_temp_dir(), 'bert-config-');
            self::assertIsString($path);
            file_put_contents($path, json_encode($data, JSON_THROW_ON_ERROR));
            try {
                try {
                    (new BertConfigReader())->read($path);
                    self::fail('Invalid BERT config accepted.');
                } catch (ModelException|\InvalidArgumentException) {
                    self::addToAssertionCount(1);
                }
            } finally {
                @unlink($path);
            }
        }
    }

    private function configFile(): string
    {
        $path = tempnam(sys_get_temp_dir(), 'bert-config-');
        self::assertIsString($path);
        file_put_contents($path, json_encode($this->config(), JSON_THROW_ON_ERROR));
        return $path;
    }

    /**
     * @param array<string, mixed> $replace
     * @return array<string, mixed>
     */
    private function config(array $replace = []): array
    {
        return array_replace([
            'architectures' => ['BertModel'], 'vocab_size' => 30522, 'hidden_size' => 384,
            'intermediate_size' => 1536, 'num_attention_heads' => 12, 'num_hidden_layers' => 12,
            'max_position_embeddings' => 512, 'type_vocab_size' => 2, 'layer_norm_eps' => 1e-12,
            'hidden_act' => 'gelu', 'position_embedding_type' => 'absolute',
        ], $replace);
    }
}
