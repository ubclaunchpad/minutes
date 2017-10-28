# minutes

[![ZenHub](https://raw.githubusercontent.com/ZenHubIO/support/master/zenhub-badge.png)](https://zenhub.com)

[![Build Status](https://travis-ci.org/ubclaunchpad/minutes.svg?branch=master)](https://travis-ci.org/ubclaunchpad/minutes)

Audio speaker diarization and transcription API.

## :running: Getting Started


### :rocket: Running the Server

```
make build-prod
make run
```

### :rainbow: Running the Research Environment

```
make build-dev
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

## :computer: Creating Training Data

We have a pipeline that is designed to take YouTube videos with
transcripts and convert them into training data. `pipeline.py`
is a CLI that will attempt to download the transcript and audio
data for a given video, as well as prompt for transcript delimiter
information that the rest of the pipeline uses to create labelled
data.

```bash
$ cd app/collector
$ ./pipeline.py <video_id>
```

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
