# Downloads
## Weather datasets
### Downloading
Download these datasets, Decompress them to a folder, and follow the layout below.

* [FWID](https://ieee-dataport.org/documents/five-class-weather-image-dataset-1)
* [Image2Weather (Image files)](https://www.cs.ccu.edu.tw/~wtchu/projects/Weather/)

```title="data/Weather/"
FWID/
    original/
        clear/
            ...
        cloudy/
            ...
        ...       # Other weathers.

Image2Weather/
    original/
        sunny/
        cloudy/
        ...       # Other weathers.
```

### Conversion and Combination

Run the command below, it will do these steps:

1. Convert the datasets to the defined format (e.g. `FWID/original/` -> `FWID/dataset/`)
2. Combine these datasets

Please fill the arguments in `<...>` first.

```bash
root="<the folder contains weather datasets>"
out="$root/FWID_Image2Weather"

conda activate uav_data_generation

bash scripts/datasets/prepare_fwid_image2weather.sh "$root" "$out"
```

For more details about the formats in the entire pipeline, see [Dataset in Design](../design/dataset.md).

## Blender scene and assets
Decompress it to a folder.

* [Google Drive (permission required)](https://drive.google.com/file/d/1YEW_SlM_R7qLyaqJdS3NMnRWrmtLFzLl/view?usp=drive_link)

For more details about the resources, see [Resources](../resources.md).
