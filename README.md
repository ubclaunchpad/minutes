# minutes

[![ZenHub](https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png)](https://zenhub.com)

[![Build Status](https://travis-ci.org/ubclaunchpad/minutes.svg?branch=master)](https://travis-ci.org/ubclaunchpad/minutes)

Audio speaker diarization and transcription library.

## Under Construction!

## :running: Development

To use our [conda](https://conda.io/docs/user-guide/install/index.html) environment,

```bash
conda env create -f environment.yml
source activate minutes
```

## Testing

```bash
pytest --cov=minutes -vvv test
```

## Building the Conda Package

Specify a new git version tag, edit the `meta.yml` and run:

```bash
conda-build .
```
