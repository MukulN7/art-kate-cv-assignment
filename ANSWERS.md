# Computer Vision / ML Engineer Assignment Answers

## Part B Diagnose 3 Broken CV Snippets

### B1: Snippet 1

#### Defects

1) The offset padding is never deducted again. When the image is placed on the canvas, it is shifted by a certain number of pixels (dw and dh) so as to be centred rather than remaining in a corner. However, when converting the box coordinates back to the original image, the code merely divides by r and never first subtracts dw and dh, meaning that each coordinate retains a portion of that shift which has been incorporated into it. Furthermore, the preprocess function doesn't pass dw and dh back to the caller at all, so there is no possibility of correcting this issue later on without first altering the function's signature. Thus, this bug is not merely due to an incorrect formula; it is the fact that the information necessary to arrive at the correct formula is not being passed on.

2) The boxes array is altered directly. Instead of creating a copy of it first, the function modifies the array that was passed to it. As a result, the original, unprocessed box values are lost the first time postprocess is run. If that array is then used elsewhere afterwards for example, for logging, for drawing debug overlays, or for a second pass of processing it will silently hold already converted and possibly incorrect numbers. Bugs resulting from this type of in place mutation are particularly frustrating to track down since the array itself provides no indication that it has been changed.

#### Why It Survives Casual Testing

In the case of a square image, both dw and dh work out as exactly 0 since no padding is required when both sides fit perfectly. There's therefore nothing to subtract, and by coincidence the erroneous line and the correct one yield the same output. The problem only appears with non square images, and even in those cases the error is by a fixed number of pixels, which merely seems like ordinary small inaccuracy rather than an obvious crash or the production of garbage data. It is easily missed altogether when one or two test images are quickly examined visually, particularly if the test images in question are square or nearly square, or if the areas that are checked happen to be near the centre of the frame where the effect of the offset is not very noticeable.

#### Corrected Code:

```python
def preprocess(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dh, dw = (size - nh) // 2, (size - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return blob[None], r, dw, dh   # now also returns the padding offsets


def postprocess(boxes, r, dw, dh, orig_shape):
    boxes = boxes.copy()  # don't touch the caller's array
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])
    return boxes
```

#### Test / Validation

Pick a box on a rectangular test image, run it through preprocess and then postprocess, and check you get back the same coordinates you started with.

```python
import numpy as np

def test_letterbox_roundtrip_nonsquare():
    orig_shape = (1080, 1920)  # rectangular, so padding will be nonzero
    size = 640
    h, w = orig_shape
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    dh, dw = (size - nh) // 2, (size - nw) // 2

    orig_box = np.array([1800, 900, 1850, 950], dtype=np.float32)
    canvas_box = orig_box.copy()
    canvas_box[[0, 2]] = orig_box[[0, 2]] * r + dw
    canvas_box[[1, 3]] = orig_box[[1, 3]] * r + dh

    recovered = postprocess(canvas_box[None, :], r, dw, dh, orig_shape)[0]
    assert np.allclose(recovered, orig_box, atol=1.0)
```

If this passes, meaning the recovered box is basically equal to the original within about a pixel, the fix works. On the old code, it fails by a fixed, nonzero amount, and that amount will match dw divided by r or dh divided by r exactly, which is a good way to confirm you've found the right bug.

### B2: Snippet 2

#### Defects:

1) When images are flipped, the original, unflipped labels are retained. The code flips the image from left to right using cv2.flip but keeps the same box coordinates as in the unflipped image. As a result, for each flipped copy the boxes now indicate the mirror image of where the object actually is. Since one out of every three of the samples appended for each image is this flipped and mislabeled version, about one third of the entire augmented dataset has boxes that are incorrect from the beginning.

2) The train and validation split takes place after the augmentation, not before. All the copies of an image whether the original or the ones that have been flipped or brightened are combined into a single large list and then shuffled as a whole, which means that it's possible for two copies that are nearly identical from the same source photo to end up on either side of the 80/20 split. In this way, the rule that no image or crop of it should appear on both the training and validation sides is violated, since that rule was intended to ensure that training and validation remain independent.

3) Since no random seed is specified, each time the script runs random.shuffle uses a different ordering. As a result, the split will vary each time the same code is rerun, which makes it impossible to reliably reproduce or compare the reported accuracy figures across different runs.

#### Why It Survives Casual Testing

The flipped label bug doesn't cause a crash or result in any error; instead, the model trains using some boxes that indicate the wrong location, and the training loss still decreases normally since the network is merely learning from these poor examples rather than failing completely. The training process appears entirely normal. The split leakage bug is more serious since it is concealed within the validation score itself. As nearly identical copies of the same photo can end up in both the training and validation sets, the model is in effect being tested on data that it has already partly memorised, which is why the validation accuracy appears to be fairly high. The issue only becomes apparent when the model is presented with genuinely new images that it has never seen in any form this is precisely the kind of collapse on real data that was described in the assignment. This is the flaw that does the most damage to the reported metric, because it doesn't just make the model slightly less effective; it makes the metric you're using to assess the model completely untrustworthy from the start.

#### Corrected Code:

```python
import glob
import random
import cv2

def flip_labels_horizontal(labels):
    """labels: array of (class, x_center, y_center, w, h), normalized 0 to 1."""
    flipped = labels.copy()
    flipped[:, 1] = 1.0 - labels[:, 1]  # mirror the x center
    return flipped

random.seed(42)  # makes the split reproducible

images = sorted(glob.glob("dataset/images/*.jpg"))
random.shuffle(images)  # shuffle the original images, before creating any augmented copies

split = int(0.8 * len(images))
train_paths, val_paths = images[:split], images[split:]

def build_augmented_set(paths):
    out = []
    for path in paths:
        img = cv2.imread(path)
        labels = load_labels(path)
        out.append((img, labels))
        out.append((cv2.flip(img, 1), flip_labels_horizontal(labels)))
        out.append((adjust_brightness(img, 1.3), labels))  # brightness doesn't move objects, so labels stay the same
    return out

train = build_augmented_set(train_paths)
val = build_augmented_set(val_paths)
```

#### Test/Validation:

```python
def test_flip_labels_horizontal():
    labels = np.array([[0, 0.2, 0.5, 0.1, 0.2]])
    flipped = flip_labels_horizontal(labels)
    assert np.isclose(flipped[0, 1], 0.8)  # x center mirrors correctly
    assert np.isclose(flipped[0, 2], labels[0, 2])  # y unaffected

def test_no_leakage_between_splits():
    assert not (set(train_paths) & set(val_paths))
```

Also worth a quick manual look: draw the flipped box on a few flipped images and confirm it still sits on the object rather than floating over empty background. This catches sign errors that a numeric test alone might not.

### B3: Snippet 3

#### Defects

1) Negative overlap is not rounded off to zero. In cases where two boxes do not in fact overlap at all, the width or height of their intersection becomes negative. Since multiplying two negative numbers yields a positive result, the program ends up showing a positive intersection area for boxes that share no pixels. This is a typical example of a situation in which the mathematics are correct on their own but silently fail when applied to the case of no overlap, a case which has not been specifically addressed.

2) The formula for area is based on the wrong box format. Although the boxes are listed as xyxy, which means x1, y1, x2, y2, the area is calculated by multiplying box[2] by box[3], as if the two values were the width and height. In fact, in the xyxy format these values are just the raw coordinates, so this calculation yields a number that has no relation to the actual size of the box, and even worse, the number increases simply depending on the position of the box in the image rather than on how large it is.

3) Although the classes argument is accepted, it is never utilized. The nms function contains nothing that reads this argument, even though it is clearly intended to separate the detections by object type. Consequently, in the suppression loop all the boxes compete with one another regardless of their class, which means that a box from one class can end up eliminating a box from a totally different class simply because the two of them overlap in space.

#### Why It Survives Casual Testing

In a crowded scene the boxes tend to be near one another, which means that even when two boxes do not actually overlap they only have very small negative gaps between them. The resulting phantom intersection remains small and seldom goes above the 0.5 level, so no unusual behaviour is apparent. It is only in a sparsely populated scene, in which a real detection is far away from all the other boxes, that the negative gap becomes large enough for the phantom intersection to push the IoU over the threshold. At this stage NMS wrongly treats the detection as a duplicate and discards it, even though it had never overlapped with anything. Because the assignment states that the total number of detections appears reasonable, a simple check such as the average number of boxes per image will entirely fail to pick up this problem. The only way to detect it is by specifically examining sparse scenes that contain isolated objects.

#### Corrected Code:
```python
def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter)


def nms(boxes, scores, classes, thr=0.5):
    keep = []
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        cls_boxes, cls_scores = boxes[idx], scores[idx]

        order = cls_scores.argsort()[::-1]
        cls_keep = []
        while order.size > 0:
            i = order[0]
            cls_keep.append(i)
            ious = iou(cls_boxes[i], cls_boxes[order[1:]])
            order = order[1:][ious < thr]

        keep.extend(idx[cls_keep].tolist())

    return keep
```

#### Test/Validation:
```python
def test_iou_zero_for_disjoint_boxes():
    box = np.array([50, 50, 60, 60], dtype=np.float32)
    boxes = np.array([[200, 200, 210, 210]], dtype=np.float32)  # far away, no overlap
    result = iou(box, boxes)
    assert result[0] == 0.0

def test_nms_is_class_aware():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10]], dtype=np.float32)  # heavily overlapping
    scores = np.array([0.9, 0.8])
    kept = nms(boxes, scores, classes=np.array([0, 1]), thr=0.5)
    assert set(kept) == {0, 1}  # different classes shouldn't suppress each other
```

The first test is the key one. On the old code, it returns a large positive number instead of 0.0 for two boxes that clearly don't touch, and that's the exact bug that makes real, isolated detections disappear in sparse frames.



## Part C Diagnose 3 Production CV Failures

### C1: Accuracy Collapse After INT8 Quantisation (0.91 to 0.58 mAP@0.5)

#### Diagnosis

A drop this large with architecture and data unchanged has to come from one of three places: the PyTorch to ONNX export before quantization even happens, a mismatch between the two evaluation harnesses, or something inside INT8 itself: bad calibration, wrong granularity, or a layer that got quantized when it shouldn't have. One measurement narrows this down fast.

#### Elimination Logic

Run the export at FP32 ONNX and FP16 ONNX before touching INT8. If FP32 ONNX matches PyTorch near 0.91, both harnesses agree and the fault is INT8 specific. If FP32 ONNX is already low, the problem predates quantization. Check the opset for silent op mismatches (SiLU, Mish) and confirm preprocessing and postprocessing match exactly between harnesses.

If FP32 holds but FP16 also collapses, dynamic range compression isn't the cause; something in the graph only breaks once TensorRT builds an engine on the Jetson. If FP16 holds and only INT8 collapses, per class AP and per layer quantization error split the remaining causes: broad uniform degradation suggests a calibration set problem; degradation concentrated in the detection head suggests a precision sensitive layer got quantized; a pattern tied to specific channels suggests per tensor quantization flattening a wide range layer. Most quantization toolchains expose these per layer numbers in the calibration report, so pulling them costs little.

#### Evidence / Measurements

mAP@0.5 and mAP@0.5:0.95 at all four stages; per class AP; confidence score histograms (quantization noise often shows as compressed confidence before misclassification); IoU distribution of matched predictions (separates loosely localized boxes from missing ones); calibration set size and composition; per layer activation range before and after calibration; model checksum and ONNX opset actually deployed on the Orin. Lining these up side by side across the four stages usually makes the failing stage more obvious than a guess.

#### Root Cause

Given the stated numbers, the most defensible hypothesis is post training quantization (PTQ) on a weak calibration set, worsened by quantizing the detection head along with everything else. Box regression is unusually sensitive to quantization noise. This is inference, not confirmation. The FP32 and FP16 numbers aren't in hand yet, so an export or decode mismatch remains just as plausible.

#### Fix

If confirmed quantization related, expand the calibration set to a few hundred images spanning the full class and scale range, switch to per channel weight quantization, and keep the detection head in FP16 while quantizing only the backbone and neck. If PTQ still can't close the gap, move to quantization aware training, since PTQ alone often has a real ceiling on architectures with wide dynamic range activations. If confirmed export or decode mismatch, align preprocessing and postprocessing exactly, pin the opset, and diff raw pre NMS outputs on a few images rather than only comparing final boxes.

#### Validation

Run the validation set through the corrected pipeline. mAP@0.5 should land within 1 to 2 points of 0.91. Confirm the FP32 ONNX checkpoint still tracks PyTorch within half a point, and measure latency on the Orin, since recovering accuracy while losing the latency benefit isn't a fix. Keep the four stage comparison as a check on every future export.



### C2: One Camera Out of Twelve Is Wrong

#### Diagnosis

Eleven cameras running identical model and code come out correct; only the twelfth doesn't, which rules out the model and shared code immediately. A consistent direction rules out network jitter or frame desync. Growth toward the edges rules out a fixed pixel offset. A magnitude scaling with distance from center points to a scale or resolution mismatch specific to that camera, or to uncorrected lens distortion. Pure radial distortion would push boxes in opposite directions on opposite edges, so a consistent shift direction favors a scale, resolution or crop mismatch, though a distortion component can't be excluded without the trace below.

#### Elimination Logic

Model and code are ruled out by the eleven cameras. Network or frame corruption is ruled out because the error is a fixed bias rather than intermittent jumping. Lighting causing the model to misjudge location is harder to dismiss, but one check settles it: compare the model's raw output box before camera specific coordinate mapping against what's displayed. If raw output is already off, the model is implicated. If raw output is correct and the offset appears only after mapping, the fault sits downstream in calibration.

From there the remaining candidates need to be told from each other, not from the model: a wrong intrinsics file applied to feed 12, maybe a silent fallback to another camera's file; a resolution mismatch between what camera 12 streams and what preprocessing assumes; or a real physical lens or mount difference unaccounted for in calibration.

#### Evidence / Measurements

Check whether camera 12's actual stream resolution matches what preprocessing assumes. Diff the calibration file ID and checksum loaded at runtime against what should be assigned. Trace a fixed landmark's pixel position through each stage (undistort, resize, model, postprocess, display mapping) to pinpoint where the shift enters. Replay the stream through a separate pipeline using a known good calibration file as a control. Log offset magnitude against distance from center to confirm it scales with distance from center, as a scale or distortion error would, rather than being a constant offset. A quick sanity check is overlaying the object's box across a few consecutive frames to confirm the drift stays fixed rather than wandering.

#### Root Cause

An offset isolated to one camera, consistent in direction, and growing with distance from center is most consistent with a mismatched calibration file or a resolution assumption that doesn't match what the camera streams. Single camera exclusivity rules out the model and code; edge scaling favors a calibration or scale error over a constant offset, and the consistent direction favors scale, resolution or crop over pure lens distortion. Which error it is isn't established without the trace above.

#### Fix

Correct whichever calibration file is assigned to camera 12, or redo calibration if the lens or mount genuinely differs. Add a startup check asserting each camera's declared resolution matches what's actually received, failing loudly instead of silently defaulting.

#### Validation

Repeat the center versus edge offset measurement on camera 12. Confirm it lands within the same tolerance band as the other eleven. Keep the resolution assertion running continuously so future drift surfaces immediately.



### C3: Silent Degradation Over Three Months (97 percent to 84 percent, No Code/Model Change)

#### Diagnosis

No code or model change, and a drift over three months that nothing reported altered, all point to something changing in the input or setup without getting logged, since line changes rarely reach whoever owns the model. It is also worth holding open a tracking issue, such as belt speed affecting frame association, rather than a true accuracy change.

#### Elimination Logic

**Camera or environmental drift** (lens fouling, seasonal lighting, a bulb swap, a shifted mount) is checked by comparing recent frames against the original period on brightness, contrast and blur, and whether the belt still occupies the same region of frame. Stable statistics rule this out entirely.

**An unreported packaging change**, a new SKU or a variant close to the trained class, shows up as a rise in low confidence detections or a cluster that doesn't match trained prototypes, visible in the confidence distribution shifting or turning bimodal.

**Tracking logic degrading** rather than detection quality: belt speed increasing and causing more blur or faster transit than the tracker's association step handles shows up as double counts or missed counts while per frame detection stays fine. This is checked through dropped frame counts, FPS over time, and whether count timing shows double counts, such as two counts registering less than a frame interval apart, or unexplained gaps.

Stable image statistics rule out the first. Shift weight to the second or third. A stable confidence distribution with abnormal count timing isolates the third. A shifted confidence distribution is the signal for the second.

#### Evidence / Measurements

Image statistics over time against baseline, confidence distribution over time per class, detection rate if multiple SKUs are involved, dropped frame count and FPS, count event timing, and any line or operations change log.

#### Root Cause

Camera drift and an unreported packaging change are the two likeliest to check first, both investigable from image and log data alone. Which one actually happened isn't settled by the scenario; this can't be diagnosed after the fact with confidence because the monitoring that would have caught it wasn't running. The fix depends on which the evidence points to.

#### Fix

If drift is confirmed, reposition or recalibrate the camera. If packaging changed, retrain or fine tune on updated imagery. If it's tracking or belt speed, retune the association thresholds.

The durable fix either way is a monitoring signal that should have existed from the start: a weekly comparison of median confidence and mean brightness and contrast against a deployment baseline, alerting when confidence drops more than about 1.5 standard deviations (measured against the baseline's week to week variation) or brightness moves outside a fixed band. Because the tracking cause wouldn't move either of those, add a count interval check to the same weekly job: the share of counts registering less than a frame interval apart, and unexplained gaps, against baseline. This needs no ground truth labels, only a stored baseline, and should make a real drift visible within one to two weeks instead of three months, provided the baseline is stable enough that a 1.5 standard deviation move is small relative to the drift; if the baseline is noisy, the threshold or window needs tuning against a few weeks of pre drift data.

#### Validation

A small labeled spot check, around 200 items, should confirm accuracy recovered to at least 97 percent or an agreed threshold. Then watch the monitoring signal for two to four weeks to confirm it stays inside the baseline band.



## Part D Status

Part D was not completed within the assignment's 24 hour submission window.
