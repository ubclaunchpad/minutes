# minutes

[![ZenHub](https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png)](https://zenhub.com)

[![Build Status](https://travis-ci.org/ubclaunchpad/minutes.svg?branch=master)](https://travis-ci.org/ubclaunchpad/minutes)

Audio speaker diarization and transcription API.

## :running: Getting Started


### Running the Server

```
make
make run
```

### Running the Research Environment

```
make
make dev
```

## :point_up: Deployment

We deploy continuously using Travis CI and a Docker Hub deploy bot. You can also deploy the production and development images manually using,

```
make push-dev
make push-prod
```

You will need to set `DOCKER_USERNAME` and `DOCKER_PASSWORD` and be a member of the `ubclaunchpad` docker hub organization to deploy manually.
