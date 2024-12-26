# Weather Anti-UAV

## Environment
* Ubuntu 22.04
* Blender 3.5.1
* NVIDIA GeForce RTX 3090

## Downloads
| Name                       | Description                                                  | transformations                                      | # of samples               | Link              |
| -------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- | -------------------------- | ----------------- |
| Weather Anti-UAV S         | All weather images are real.                                 | Image Harmonization (PIH)                            | 10000 / 5000 (val)         | [Google Drive](https://drive.google.com/file/d/14f9Nw60Bp_lfPUADRXK8RrroWRrnNRgv/view?usp=drive_link)  |
| Weather Anti-UAV S Fake    | All weather images are transformed from clear/cloudy images. | Image Harmonization (PIH), Weather Stylization (CUT) | 10005 / 4995 (val)         | [Google Drive](https://drive.google.com/file/d/1ylnsPlJOSlrrtm0QO0mbNP8T4XpsdIh_/view?usp=drive_link)  |

## Blender configs
### Weather Anti-UAV S
```python title="configs/weather_anti_uav_s.py" linenums="1"
--8<-- "configs/weather_anti_uav_s.py"
```

### Weather Anti-UAV S Fake
```python title="configs/weather_anti_uav_s_fake.py" linenums="1"
--8<-- "configs/weather_anti_uav_s_fake.py"
```
