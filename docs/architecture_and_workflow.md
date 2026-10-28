# System Architecture and End-to-End Workflow

These diagrams describe the implemented data/model path and local application.
The browser sends an uploaded image only to the local Python server; the server
calls the existing combined inference function and existing checkpoint-backed
prediction APIs.

## System architecture

```mermaid
flowchart LR
    User[Research user]
    Browser[Web browser<br/>HTML CSS JavaScript]
    Server[Local Python HTTP server<br/>upload validation and response]
    Combined[Combined inference<br/>predict_mri]
    Classifier[Saved CNN classifier<br/>checkpoint preprocessing]
    Segmenter[Saved U-Net segmenter<br/>checkpoint preprocessing]
    Device[Device selector<br/>MPS or CPU]
    Original[Original MRI<br/>RGB view]
    ClassOut[Predicted class<br/>four probabilities]
    SegOut[Probability map<br/>fixed threshold 0.5]
    Align[Nearest-neighbor mask alignment<br/>overlay on original]
    Artifacts[Existing experiment artifacts<br/>checkpoints and configs]

    User --> Browser
    Browser -->|multipart upload| Server
    Server -->|validated temporary image| Combined
    Artifacts --> Classifier
    Artifacts --> Segmenter
    Artifacts --> Combined
    Combined --> Device
    Device --> Classifier
    Device --> Segmenter
    Classifier --> ClassOut
    Segmenter --> SegOut
    SegOut --> Align
    Combined --> Original
    Original --> Align
    Original --> Server
    ClassOut --> Server
    SegOut --> Server
    Align --> Server
    Server -->|JSON and image data URLs| Browser
    Browser -->|display and downloads| User
```

The classifier and segmenter preprocess inputs independently using settings
saved with their respective checkpoints. The HTTP service validates supported
image types, file size and readability; validates output class-probability
mapping, configured threshold and image dimensions; and responds with
user-facing errors without returning stack traces or local paths.

## End-to-end data, training, and evaluation workflow

```mermaid
flowchart TD
    RawClass[Brain Tumor MRI<br/>image labels]
    RawBRISC[BRISC 2025<br/>manifest-linked MRI and masks]
    Inspect[Dataset inspection<br/>readability and source inventory]
    Audit[SHA-256 leakage audit<br/>derived manifests]
    Group[Duplicate groups assigned together<br/>flag conflicting mask groups]
    Split[Derived training / validation / test partitions]
    Preprocess[MRI resize and intensity scaling<br/>mask threshold 128 and nearest resize]
    TrainClass[Train compact CNN<br/>select by validation macro-F1]
    TrainUNet[Train U-Net<br/>select by validation mean Dice]
    CkptClass[Selected classifier checkpoint]
    CkptUNet[Selected U-Net checkpoint]
    TestClass[Separate derived-test classification evaluation]
    TestSeg[Separate derived-test segmentation evaluation]
    App[Local inference application<br/>class + mask + overlay]
    Report[Saved metrics, figures,<br/>manifests and reports]

    RawClass --> Inspect
    RawBRISC --> Inspect
    Inspect --> Audit
    Audit --> Group
    Group --> Split
    Split --> Preprocess
    Preprocess --> TrainClass
    Preprocess --> TrainUNet
    TrainClass -->|validation selection| CkptClass
    TrainUNet -->|validation selection| CkptUNet
    CkptClass --> TestClass
    CkptUNet --> TestSeg
    CkptClass --> App
    CkptUNet --> App
    TestClass --> Report
    TestSeg --> Report
    Audit --> Report
    Preprocess --> Report
```

The test partitions are held out for the separate evaluation phase and are not
used to select model checkpoints. Classification uses only labeled
classification images. Segmentation uses BRISC image-mask pairs and excludes
the 46 flagged conflicting-mask rows. Pixel-level test metrics and class
classification metrics are reported separately in
[the experiment report](methodology_and_results.md).
