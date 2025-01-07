# Overview

<figure markdown="span">
  ![Overview of generation pipeline](assets/overview.png){ style="border-radius:10px;" }
  <figcaption>Overview of generation pipeline</figcaption>
</figure>

## Goal
Aims to propose a generation pipeline for UAV dataset to collect adverse weather images.

## Features
* Realistic dataset generation
* Support various UAV properties randomization, e.g. geometry, etc.
* Support motion blur, overlap control among UAV samples, etc.
* Transfer weather without simulation
* Modular design, easy to support additional processing
* Use non-PBR/PBR textures (**CC0**, **RF**) and open/closed-source models

## Sections
* [Getting started](getting-started): A quick guide to generate an UAV dataset.
* [User guides](user-guides/customization.md): Tutorials about configuration or customization, etc.
* [Presets](presets/weather-anti-uav.md): Several generated datasets.
* [Design](design/dataset.md): Details of the system design of generation pipeline.
* [Resources](resources.md): Utilized resources and useful references in the generation pipeline.
