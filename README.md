# minutes

[![ZenHub](https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png)](https://zenhub.com)

[![Build Status](https://travis-ci.org/ubclaunchpad/minutes.svg?branch=master)](https://travis-ci.org/ubclaunchpad/minutes)

Audio speaker diarization and transcription API.

## :running: Getting Started


### :rocket: Running the Server

```
make
make run
```

### :rainbow: Running the Research Environment

```
make dev
```

## :point_up: Deployment

We deploy continuously using Travis CI and a Docker Hub deploy bot. 

You can also deploy the production and development images manually if you like.

```
make push-dev
make push-prod
```

You will need to set `DOCKER_USERNAME` and `DOCKER_PASSWORD` and be a member of the `ubclaunchpad` docker hub organization to deploy manually.


## :point_right: Pushing Training Data

You can push training data into the research environment on DigitalOcean. 
You will need to collect the instance PEM file from your tech lead. Place
the PEM locally in `~/.ssh/id_minutes`. Set the environment variable 
`MINUTES_RESEARCH_INSTANCE` in your environment to the IP address of the
DigitalOcean instance (available on Slack or from your tech lead).

Then, if you wish to push the file `bigdata.csv`, use the following command:

```bash
make FILE=bigdata.csv push-data
```

It will appear in the `data` folder on the research platform.