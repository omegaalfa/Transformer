dnl PHP Transformer extension scaffolding.
dnl Build targets will be introduced only after the native runtime ABI exists.
PHP_ARG_ENABLE([transformer],
  [whether to enable transformer support],
  [AS_HELP_STRING([--enable-transformer], [Enable transformer scaffolding])],
  [no])
