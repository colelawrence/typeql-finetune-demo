/** @type {import('stylelint').Config} */
module.exports = {
  extends: ['stylelint-config-standard'],
  plugins: ['stylelint-declaration-strict-value'],
  rules: {
    // ════════════════════════════════════════════
    // TOKEN ENFORCEMENT
    // ════════════════════════════════════════════

    // Block raw colors everywhere except token definitions
    'color-no-hex': true,
    'color-named': 'never',
    'function-disallowed-list': ['rgb', 'rgba', 'hsl', 'hsla', 'hwb', 'lab', 'lch'],

    // Require var(--*) for key design properties
    'scale-unlimited/declaration-strict-value': [
      [
        // Colors
        'color', 'background-color', 'background',
        'border-color', 'border-top-color', 'border-right-color',
        'border-bottom-color', 'border-left-color',
        'outline-color', 'fill', 'stroke',
        // Typography
        'font-size', 'line-height', 'letter-spacing', 'font-family', 'font-weight',
        // Radii
        'border-radius', 'border-top-left-radius', 'border-top-right-radius',
        'border-bottom-right-radius', 'border-bottom-left-radius',
      ],
      {
        disableFix: true,
        ignoreValues: [
          'inherit', 'initial', 'unset', 'revert', 'currentColor',
          'transparent', 'none', 'auto', 'normal', '0', '50%',
        ],
        message: 'Use a CSS token (var(--*)). Raw values are banned. (scale-unlimited/declaration-strict-value)',
      },
    ],

    // Ban private tokens (--_*) outside token definitions
    'declaration-property-value-disallowed-list': [
      { '/.*/': ['/var\\(\\s*--_/'] },
      { message: 'Private tokens (--_*) are for definitions only. Use semantic tokens (--color-*, --space-*, etc.).' },
    ],

    // Ban raw units in tokenized properties
    'declaration-property-unit-disallowed-list': {
      'font-size': ['px', 'pt'],
      'border-radius': ['px', 'rem', 'em'],
    },

    // ════════════════════════════════════════════
    // QUALITY
    // ════════════════════════════════════════════
    'declaration-no-important': true,

    // ════════════════════════════════════════════
    // RELAX FORMATTING (don't fight the style)
    // ════════════════════════════════════════════
    'value-keyword-case': null,
    'selector-class-pattern': null,
    'declaration-empty-line-before': null,
    'comment-empty-line-before': null,
    'custom-property-empty-line-before': null,
    'no-descending-specificity': null,
    'color-hex-length': null,
    'color-function-notation': null,
    'alpha-value-notation': null,
    'media-feature-range-notation': null,
    'custom-property-pattern': null,
    'shorthand-property-no-redundant-values': null,
    'declaration-block-no-redundant-longhand-properties': null,
    'declaration-block-single-line-max-declarations': null,
    'rule-empty-line-before': null,
    'keyframes-name-pattern': null,
    'property-no-vendor-prefix': null,
    'lightness-notation': null,
    'hue-degree-notation': null,
  },
  overrides: [
    {
      // Token definition file — raw values are allowed here (it defines the tokens)
      files: ['ui/styles.css'],
      rules: {
        // The :root block defines primitives with oklch() — allow it
        // We can't scope to just :root, so we exempt the whole file and
        // rely on the private token ban (--_*) to catch misuse elsewhere
        'color-no-hex': null,
        'function-disallowed-list': null,
        'scale-unlimited/declaration-strict-value': null,
        'declaration-property-unit-disallowed-list': null,
        'declaration-property-value-disallowed-list': null,
      },
    },
  ],
  ignoreFiles: ['**/node_modules/**'],
};
