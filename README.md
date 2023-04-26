# PointVisualizaiton
A point cloud visualization repo

- **Multiple Point Cloud Renderer using Mitsuba 3**
- **Real Time Point Cloud View Tool**

![figure](figure.jpg)

## Dependencies

python >= 3.7
```bash
pip install -r requirements.txt
```

## Usage

```bash
# Render a single file to colorful image
python main.py --path <file path> --render

# Render a single file to knn cluster colorful image
python main.py --path <file path> --render --knn

# Render a single file to white image
python main.py --path <file path> --render --white

# Render a single file to split part
python main.py --path <file path> --part

# Render a single file with rotation 90 degree in y axis
python main.py --path <file path> --render --rot 0 90 0

# view real time point cloud
python main.py --path <file path> --tool
```



## Source

Many thanks to following codes that help us a lot in building this codebase:

* [PointFlowRenderer](https://github.com/zekunhao1995/PointFlowRenderer)
* [Mitsuba2PointCloudRenderer](https://github.com/tolgabirdal/Mitsuba2PointCloudRenderer) 
* [PointSetGeneration](https://github.com/fanhqme/PointSetGeneration)
