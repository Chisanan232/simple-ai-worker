# Reusable Workflows Guide

This template provides a comprehensive set of reusable GitHub Actions workflows that can be called from other repositories. These workflows are designed to provide standardized CI/CD operations for Python projects using UV.

## Overview

All reusable workflows are prefixed with `rw_` and can be called from external repositories using the format:

```yaml
uses: Chisanan232/Template-Python-UV-Project/.github/workflows/rw_<workflow_name>.yaml@master
```

## Reusable Actions

In addition to workflows, this template also provides **reusable actions** that can be called from external repositories. Actions provide reusable functionality that can be used across multiple workflow steps.

### Available Actions

#### `setup-python-uv`
A composite action that sets up Python, installs UV, and optionally installs project dependencies with caching support.

**Inputs:**
- `python-version` (string, optional): Python version to use (default: '3.13')
- `install-dependencies` (boolean, optional): Whether to install project dependencies (default: 'true')
- `dependency-groups` (string, optional): Dependency groups to install (default: 'release-ci')
- `cache-dependencies` (boolean, optional): Whether to cache UV dependencies (default: 'true')

**Usage:**
```yaml
steps:
  - name: Checkout
    uses: actions/checkout@v5
  
  - name: Setup Python with UV
    uses: Chisanan232/Template-Python-UV-Project/.github/actions/setup-python-uv@master
    with:
      python-version: '3.12'
      install-dependencies: 'true'
      dependency-groups: 'dev,test'
      cache-dependencies: 'true'
```

**Features:**
- Automatically sets up Python with the specified version
- Installs and configures UV package manager
- Intelligent caching of UV dependencies for faster builds
- Flexible dependency group installation
- Verification steps to ensure proper setup

## Available Reusable Workflows

### Core Testing & CI Workflows

#### 1. `rw_build_and_test.yaml`
Runs comprehensive test suite including unit, integration, e2e, and contract tests.

**Inputs:**
- `run_e2e` (boolean, optional): Whether to run end-to-end tests (default: false)

**Secrets:**
- `e2e_test_api_token` (optional): API token for e2e tests

**Usage:**
```yaml
jobs:
  test:
    uses: Chisanan232/Template-Python-UV-Project/.github/workflows/rw_build_and_test.yaml@master
    with:
      run_e2e: true
    secrets:
      e2e_test_api_token: ${{ secrets.YOUR_API_TOKEN }}
```

#### 2. `rw_run_all_test_and_record.yaml`
Orchestrates all testing and uploads coverage reports to CodeCov and SonarCloud.

**Inputs:**
- `run_e2e` (boolean, optional): Whether to run end-to-end tests (default: false)

**Secrets:**
- `codecov_token` (optional): CodeCov token for uploading coverage
- `sonar_token` (required): SonarCloud token for code quality analysis
- `e2e_test_api_token` (optional): API token for e2e tests

**Usage:**
```yaml
jobs:
  ci:
    uses: Chisanan232/Template-Python-UV-Project/.github/workflows/rw_run_all_test_and_record.yaml@master
    secrets:
      codecov_token: ${{ secrets.CODECOV_TOKEN }}
      sonar_token: ${{ secrets.SONAR_TOKEN }}
```

#### 3. `rw_uv_run_test_with_multi_py_versions.yaml`
Runs tests across multiple Python versions using UV.

**Inputs:**
- `test_type` (string, required): Type of test (unit-test, integration-test, e2e-test, etc.)
- `test_folder` (string, required): Path to test folder
- `install_dependency_with_group` (string, optional): UV dependency group to install
- `max-parallel` (number, optional): Maximum parallel jobs

**Secrets:**
- `e2e_test_api_token` (optional): API token for e2e tests

### Release & Publishing Workflows

#### 4. `rw_parse_release_intent.yaml`
Parses release configuration and determines what components should be released.

**Inputs:**
- `level` (string, optional): Release level (patch, minor, major, auto)
- `python` (string, optional): Python package release (auto, force, skip)
- `docker` (string, optional): Docker image release (auto, force, skip)
- `docs` (string, optional): Documentation versioning (auto, force, skip)
- `notes` (string, optional): Release notes

**Outputs:**
- `do_release`: Whether to proceed with release
- `level`: Determined release level
- `python`: Python release mode
- `docker`: Docker release mode
- `docs`: Documentation release mode

#### 5. `rw_build_git-tag_and_create_github-release_v2.yaml`
Creates Git tags and GitHub releases.

**Inputs:**
- `version` (string, required): Version to tag and release
- `checkout-sha` (string, optional): Specific commit SHA to checkout
- `debug_mode` (boolean, optional): Enable debug mode

**Secrets:**
- `github_auth_token` (required): GitHub token for creating releases

#### 6. `rw_python_package.yaml`
Handles Python package operations (build, test, publish).

**Inputs:**
- `operation` (string, required): Operation (build, test, publish-pypi, publish-testpypi)
- `python-version` (string, optional): Python version (default: 3.13)
- `version` (string, optional): Package version for display
- `checkout-sha` (string, optional): Specific commit SHA to checkout
- `upload-artifacts` (boolean, optional): Whether to upload artifacts (default: true)
- `artifact-name` (string, optional): Name for build artifacts

**Usage:**
```yaml
jobs:
  release-python:
    uses: Chisanan232/Template-Python-UV-Project/.github/workflows/rw_python_package.yaml@master
    with:
      operation: 'publish-pypi'
      version: '1.0.0'
      artifact-name: 'my-package'
```

### Docker Operations

#### 7. `rw_docker_operations.yaml`
Comprehensive Docker operations including build, test, push, and security scanning.

**Inputs:**
- `operation` (string, required): Operation (build, test, push, security-scan)
- `image-name` (string, optional): Docker image name
- `registry` (string, optional): Container registry (default: ghcr.io)
- `version` (string, optional): Image version/tag (default: latest)
- `platforms` (string, optional): Target platforms (default: linux/amd64,linux/arm64)
- `enable-sbom` (boolean, optional): Generate SBOM (default: false)
- `enable-signing` (boolean, optional): Sign with cosign (default: false)
- `health-check-port` (string, optional): Port for health check (default: 8000)
- `health-check-path` (string, optional): Health check endpoint (default: /health)

**Secrets:**
- `dockerhub-user` (optional): DockerHub username
- `registry-token` (optional): Registry authentication token

**Usage:**
```yaml
jobs:
  docker-release:
    uses: Chisanan232/Template-Python-UV-Project/.github/workflows/rw_docker_operations.yaml@master
    with:
      operation: 'push'
      registry: 'docker.io'
      version: '1.0.0'
      enable-sbom: true
      enable-signing: true
    secrets:
      dockerhub-user: ${{ secrets.DOCKERHUB_USERNAME }}
      registry-token: ${{ secrets.DOCKERHUB_TOKEN }}
```

### Documentation

#### 8. `rw_docs_operations.yaml`
Handles documentation operations including building, versioning, and deployment.

**Inputs:**
- `operation` (string, required): Operation (build, test, version, deploy-preview)
- `version` (string, optional): Documentation version
- `checkout-sha` (string, optional): Specific commit SHA to checkout
- `sections` (string, optional): JSON array of documentation sections
- `strategy` (string, optional): Versioning strategy
- `upload-artifacts` (boolean, optional): Whether to upload artifacts
- `commit-changes` (boolean, optional): Whether to commit changes

#### 9. `rw_organize_test_cov_reports.yaml`
Organizes and uploads test coverage reports to CodeCov.

**Inputs:**
- `test_type` (string, required): Type of test for coverage report

## Quick Start Examples

### Basic CI Workflow
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    uses: Chisanan232/Template-Python-UV-Project/.github/workflows/rw_run_all_test_and_record.yaml@master
    secrets:
      codecov_token: ${{ secrets.CODECOV_TOKEN }}
      sonar_token: ${{ secrets.SONAR_TOKEN }}
```

### Complete Release Workflow
```yaml
name: Release
on:
  workflow_dispatch:
    inputs:
      level:
        type: choice
        options: [auto, patch, minor, major]

permissions:
  contents: write
  packages: write
  id-token: write

jobs:
  intent:
    uses: Chisanan232/Template-Python-UV-Project/.github/workflows/rw_parse_release_intent.yaml@master
    with:
      level: ${{ inputs.level }}

  release-python:
    uses: Chisanan232/Template-Python-UV-Project/.github/workflows/rw_python_package.yaml@master
    needs: intent
    if: needs.intent.outputs.do_release == 'true'
    with:
      operation: 'publish-pypi'
```

## Configuration Requirements

### Required Secrets
- `CODECOV_TOKEN`: For uploading test coverage to CodeCov
- `SONAR_TOKEN`: For SonarCloud code quality analysis
- `DOCKERHUB_USERNAME` & `DOCKERHUB_TOKEN`: For DockerHub releases
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

### Project Structure Requirements
Your project should follow this structure for optimal compatibility:
```
your-project/
├── src/your_package_name/     # Your Python package
├── test/
│   ├── unit_test/
│   ├── integration_test/
│   ├── e2e_test/
│   └── contract_test/
├── scripts/
│   └── run_all_tests.sh
├── pyproject.toml
├── uv.lock
├── .coveragerc
├── pytest.ini
└── codecov.yml
```

### Customization Points
When using these workflows, remember to update:
1. Branch names (replace `<your_base_branch>` with your main branch)
2. Package names (replace `<your_package_name>` with your actual package)
3. Secret names (replace `<YOUR_*_SECRET>` with your actual secret names)
4. Path patterns in trigger conditions

## Best Practices

1. **Version Pinning**: Always use `@master` or a specific version tag when calling reusable workflows
2. **Secret Management**: Use GitHub repository secrets for sensitive data
3. **Path Filters**: Customize path filters in triggers to match your project structure
4. **Testing Strategy**: Use `run_e2e: false` for faster CI, enable for comprehensive testing
5. **Release Automation**: Use the release workflows for consistent versioning and publishing

## Troubleshooting

### Common Issues

1. **Missing Secrets**: Ensure all required secrets are configured in your repository settings
2. **Path Mismatches**: Update file paths in workflow triggers to match your project structure
3. **Permission Errors**: Verify that your repository has the correct permissions set
4. **UV Dependencies**: Ensure your `pyproject.toml` includes the necessary dependency groups

### Getting Help

For issues with the reusable workflows:
1. Check the workflow run logs for detailed error messages
2. Verify your project structure matches the expected format
3. Ensure all required secrets and permissions are configured
4. Review the example workflows in the `examples/` directory
