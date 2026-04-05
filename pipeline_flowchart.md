# Terrain Traversability Classifier Pipeline

The following flowchart breaks down the entire lifecycle of our machine learning pipeline, mapping how raw unstructured terrain data becomes a workable navigation costmap for our rover system.

```mermaid
graph TD
    %% Define Styles
    classDef dataFill fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#e2e8f0
    classDef scriptFill fill:#3182ce,stroke:#2b6cb0,stroke-width:2px,color:#fff
    classDef artifactFill fill:#48bb78,stroke:#38a169,stroke-width:2px,color:#fff
    classDef hardwareFill fill:#d69e2e,stroke:#b7791f,stroke-width:2px,color:#fff

    %% Source Data Nodes
    src_images[Raw RUGD Images]:::dataFill
    src_annos[Pixel-Level Annotations]:::dataFill
    
    %% Preprocessing
    subgraph Phase 1: Data Preprocessing
        pre_script(data_preprocessor.py):::scriptFill
        pre_extract{1D Color Dominant \nExtraction Logic}:::dataFill
        out_dataset[(Structured Patch Dataset \n13,366 128x128 patches)]:::artifactFill
    end
    
    src_images --> pre_script
    src_annos --> pre_script
    pre_script --> pre_extract
    pre_extract --> out_dataset

    %% Training
    subgraph Phase 2: CNN Training & Validation
        aug_process[Transforms: Jittering, \nFlips, Rotations]:::dataFill
        train_script(terrain_classifier.py):::scriptFill
        model_arch((Custom TerrainCNN \n4-Block Conv2D)):::dataFill
        model_weights{Trained Weights \nresults/terrain_cnn.pth}:::artifactFill
        eval_metrics[Metrics & Plots: \n95.9% Acc, 97.6% Trav Acc]:::artifactFill
    end
    
    out_dataset --> aug_process
    aug_process --> train_script
    train_script --> model_arch
    model_arch -- Optimal Path --> model_weights
    train_script --> eval_metrics

    %% Inference
    subgraph Phase 3: Traversability Inference
        inf_script(inference_costmap.py):::scriptFill
        sliding_win[Sliding Window \n128x128 Patching]:::dataFill
        score_mapper{Traversability Mapper \nSafe=0, Risky=1, Avoid=2}:::dataFill
        final_overlay[Costmap Heatmap Overlay]:::artifactFill
    end
    
    model_weights --> inf_script
    src_images -- Test Frames --> inf_script
    inf_script --> sliding_win
    sliding_win --> score_mapper
    score_mapper --> final_overlay

    %% Deployment Target
    ros2_node[[ROS 2 Nav2 Costmap \nNVIDIA Jetson / ZED 2i]]:::hardwareFill
    final_overlay -. Hardware Deployment .-> ros2_node
```

### Breakdown of Pipeline Phases
1. **Phase 1: Data Preprocessing**: We feed the raw, full-scale RUGD imagery paired with masking annotations into `data_preprocessor.py`. The script uses intensive color comparisons to chop the images down into perfectly square 128x128 patches that serve as target datasets strictly categorizing 10 terrain environments.
2. **Phase 2: CNN Training**: These patches are sent into `terrain_classifier.py`. PyTorch injects random perturbations directly mimicking environmental light variances. The resulting Custom `TerrainCNN` architecture outputs weights and visual evaluation plots.
3. **Phase 3: Live Inference**: Utilizing the trained `TerrainCNN` weights, our `inference_costmap.py` runs an automated sliding window across brand-new, unseen full-sized environment frames. It maps the classes directly out to our numeric `0/1/2` safety values, culminating as a visual heatmap directly compatible with Nav2.
