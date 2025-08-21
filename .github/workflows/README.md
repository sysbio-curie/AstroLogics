# GitHub Actions Workflows for AstroLogics

This repository contains three GitHub Actions workflows for automated building and deployment:

## Workflows

### 1. PyPI Package (`pypi-publish.yml`)
- **Triggers**: Git tags starting with `v*`, releases, manual dispatch
- **Features**:
  - Builds Python wheel and source distribution
  - Tests installation across multiple OS and Python versions
  - Publishes to PyPI on releases/tags
- **Required Secrets**: `PYPI_API_TOKEN`

### 2. Conda Package (`conda-publish.yml`)
- **Triggers**: Git tags starting with `v*`, releases, manual dispatch
- **Features**:
  - Builds conda packages for multiple platforms
  - Tests installation across multiple OS and Python versions
  - Publishes to Anaconda Cloud on releases/tags
- **Required Secrets**: `ANACONDA_TOKEN`

### 3. Docker Image (`docker-publish.yml`)
- **Triggers**: Git tags, pushes to main/master, releases, manual dispatch
- **Features**:
  - Builds multi-platform Docker images (linux/amd64, linux/arm64)
  - Publishes to GitHub Container Registry
  - Creates a Dockerfile if one doesn't exist
  - Tests the built image
- **Required Secrets**: 
  - `GITHUB_TOKEN` (automatically provided)
  - Optional: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` for Docker Hub description updates

## Setup Instructions

### 1. PyPI Deployment
1. Create an API token at https://pypi.org/manage/account/token/
2. Add the token as a repository secret named `PYPI_API_TOKEN`

### 2. Conda Deployment
1. Create an account at https://anaconda.org/
2. Generate an API token in your account settings
3. Add the token as a repository secret named `ANACONDA_TOKEN`

### 3. Docker Deployment
- GitHub Container Registry is used by default (no additional setup required)
- For Docker Hub description updates (optional):
  1. Add your Docker Hub username as `DOCKERHUB_USERNAME`
  2. Add your Docker Hub token as `DOCKERHUB_TOKEN`

## Usage

### Automatic Deployment
1. Create a git tag with version number: `git tag v0.3.1`
2. Push the tag: `git push origin v0.3.1`
3. All three workflows will automatically build and deploy

### Manual Deployment
1. Go to the "Actions" tab in your GitHub repository
2. Select the workflow you want to run
3. Click "Run workflow"

## Version Management

The workflows expect:
- Version tags in the format `vX.Y.Z` (e.g., `v0.3.0`)
- Version information in `pyproject.toml` for PyPI
- Version information in `conda/meta.yaml` for Conda

## Testing

Each workflow includes testing phases:
- **PyPI**: Tests installation across multiple OS/Python combinations
- **Conda**: Tests conda package installation and import
- **Docker**: Tests image build and package import

## Notes

- The Docker workflow automatically creates a basic Dockerfile if one doesn't exist
- All workflows support manual triggering for testing purposes
- Conda packages are built for multiple platforms (Linux, Windows, macOS)
- Docker images are built for multiple architectures (AMD64, ARM64)
