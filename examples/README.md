# Example Code to Understand How to Use Arachne APIs

You can run each script by

```sh
python3 run_tflite.py
```

## Requirements

You have to
* install all required packages and libraries ahead.
    * e.g., `tvm` and `tflite` packages should be installed to run `run_tflite.py`
* setup the rpc enviroment to use edge devices.

<!--
## Output Example
<details>
<summary>Output example running on Jetson Xavier NX</summary>

```
Node Name                            Ops                                  Time(us)  Time(%)  Shape          Inputs  Outputs
---------                            ---                                  --------  -------  -----          ------  -------
tensorrt_0                           tensorrt_0                           7239.27   77.228   (1, 91, 1917)  1       2
tensorrt_0                           tensorrt_0                           7239.27   77.228   (1, 1917, 4)   1       2
fused_vision_multibox_transform_loc  fused_vision_multibox_transform_loc  946.83    10.101   (1, 1917, 6)   3       2
fused_vision_multibox_transform_loc  fused_vision_multibox_transform_loc  946.83    10.101   (1,)           3       2
tensorrt_97                          tensorrt_97                          318.178   3.394    (1, 10, 4)     4       1
tensorrt_95                          tensorrt_95                          279.617   2.983    (1, 7668)      4       1
tensorrt_96                          tensorrt_96                          149.562   1.596    (1, 10, 6)     1       1
tensorrt_98                          tensorrt_98                          120.618   1.287    (1, 10)        1       1
tensorrt_99                          tensorrt_99                          108.717   1.16     (1, 10)        1       1
fused_vision_non_max_suppression     fused_vision_non_max_suppression     87.546    0.934    (1, 1917, 6)   4       1
fused_split_2                        fused_split_2                        50.272    0.536    (1, 1917, 1)   2       4
fused_split_2                        fused_split_2                        50.272    0.536    (1, 1917, 1)   2       4
fused_split_2                        fused_split_2                        50.272    0.536    (1, 1917, 1)   2       4
fused_split_2                        fused_split_2                        50.272    0.536    (1, 1917, 1)   2       4
fused_vision_get_valid_counts        fused_vision_get_valid_counts        38.288    0.408    (1,)           2       3
fused_vision_get_valid_counts        fused_vision_get_valid_counts        38.288    0.408    (1, 1917, 6)   2       3
fused_vision_get_valid_counts        fused_vision_get_valid_counts        38.288    0.408    (1, 1917)      2       3
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
Total_time                           -                                    9373.938  -        -              -       -
Execution time summary:
 mean (s)   max (s)    min (s)    std (s)
 0.00881    0.00918    0.00864    0.00016
```
</details>
-->
