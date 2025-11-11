# Tests

This directory is reserved for unit tests and integration tests.

## Structure

```
tests/
├── unit/           # Unit tests for individual modules
├── integration/    # Integration tests for model pipelines
└── fixtures/       # Test data and fixtures
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=model_loaders --cov=utils --cov=ui_components
```

## Test Coverage Goals

- Model loaders: prediction functions, error handling
- Utils: preprocessing, feature engineering, translation
- UI components: component rendering, data validation
