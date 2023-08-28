#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2
import numpy as np
import pmt
import sys
from pathlib import Path



try:
    from gnuradio import gr  # pytype: disable=import-error
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


class terminal_sink(gr.sync_block):
    def __init__(self, input_vlen, batch_size):
        self.input_vlen = input_vlen
        self.batch_size = batch_size
        gr.sync_block.__init__(
            self,
            name="terminal_sink",
            in_sig=[(np.float32, self.input_vlen)],
            out_sig=None,
        )
        self.batch_ctr = 0

    def work(self, input_items, output_items):
        in0 = input_items[0]
        batch = in0.reshape(self.batch_size, -1)
        self.batch_ctr += 1
        return len(input_items[0])


class yolo_bbox(gr.sync_block):
    def __init__(
        self,
        n_fft,
        n_time,
        image_shape,
        prediction_shape,
        batch_size,
        sample_rate,
        output_dir,
        confidence_threshold=0.25,
        nms_threshold=0.7,
    ):
        self.n_fft = n_fft
        self.n_time = n_time
        self.image_shape = image_shape
        self.image_vlen = np.prod(image_shape)
        self.prediction_shape = prediction_shape
        self.prediction_vlen = np.prod(prediction_shape)
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        gr.sync_block.__init__(
            self,
            name="yolo_bbox",
            in_sig=[(np.float32, self.image_vlen), (np.float32, self.prediction_vlen)],
            out_sig=None,
        )

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        # label = f'{CLASSES[class_id]} ({confidence:.2f})'
        # label = f'{class_id} ({confidence:.2f})'
        label = f"{class_id}"
        color = (255, 255, 255)  # self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(
            img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    def work(self, input_items, output_items):
        # images = input_items[0]
        # predictions = input_items[1]
        rx_times = [
            sum(pmt.to_python(rx_time_pmt.value))
            for rx_time_pmt in self.get_tags_in_window(
                0, 0, 1, pmt.to_pmt("rx_time")
            )
        ]
        rx_freqs = [
            pmt.to_python(rx_freq_pmt.value)
            for rx_freq_pmt in self.get_tags_in_window(
                0, 0, 1, pmt.to_pmt("rx_freq")
            )
        ]

        sample_start_time = rx_times[0]
        sample_end_time = rx_times[0] + (self.n_fft * self.n_time / self.sample_rate)
        time_space = np.linspace(start=sample_end_time, stop=sample_start_time, num=int(self.n_time))
        freq_center = rx_freqs[0]
        min_freq = freq_center - (self.sample_rate/2)
        max_freq = freq_center + (self.sample_rate/2)
        freq_space = np.linspace(start=min_freq, stop=max_freq, num=int(self.n_fft))
        
        # for rx_time_pmt in self.get_tags_in_window(0, 0, 1, pmt.to_pmt("rx_time")):
        #     print(f"\n{rx_time_pmt.offset=}\n")
        # print(f"\n{self.nitems_read(0)=}\n")
        # print(f"\n{len(input_items)=}\n")
        # print(f"\n{len(input_items[0])=}\n")
        # print(f"\n{rx_times=}\n")
        # print(f"\n{rx_freqs=}\n")
        # print(f"\n{input_items[0].shape=}\n")
        # print(f"\n{input_items[1].shape=}\n")

        
        image = input_items[0][0]
        image = image.reshape(self.image_shape)
        print(f"{np.min(image)=}, {np.max(image)=}")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(f"{np.min(image)=}, {np.max(image)=}")
        original_image = image
        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / 640

        print(f"{height=}, {width=}")
        print(f"{length=}")
        print(f"{scale=}")

        # .pt, yolo
        pt_model = YOLO("/home/ltindall/mini2_snr.pt")
        pt_outputs = pt_model.predict(image, conf=0.25, iou=0.7, max_det=8400, save=True, project=self.output_dir, name="predict")
        pt_outputs = pt_model.predict(image, conf=0.25, iou=0.7, max_det=8400)
        pt_outputs = pt_outputs[0].boxes.data.numpy()
        pt_outputs = np.expand_dims(pt_outputs, axis=0)
        #print(f"{pt_outputs.boxes=}")
        #print(f"{type(pt_outputs.boxes)=}")
        #pt_outputs = np.array(cv2.transpose(pt_outputs.boxes[0]))

        # .onnx, cv2
        onnx_model_file = "/home/ltindall/onnx/mini2_snr.onnx"
        onnx_model = cv2.dnn.readNetFromONNX(onnx_model_file)
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        onnx_model.setInput(blob)
        onnx_outputs = onnx_model.forward()
        onnx_outputs = np.array([cv2.transpose(onnx_outputs[0])])

        # .plan, gr-wavelearner inference
        prediction = input_items[1][0]
        prediction = prediction.reshape(self.prediction_shape)
        prediction = np.array([cv2.transpose(prediction[0])])

        # verify inference outputs 
        plan_outputs = prediction 
        print(f"{pt_outputs.shape=}")
        print(f"{onnx_outputs.shape=}")
        print(f"{plan_outputs.shape=}")
        print(f"{np.array_equal(onnx_outputs, pt_outputs)=}")
        print(f"{np.array_equal(pt_outputs, plan_outputs)=}")
        print(f"{np.array_equal(onnx_outputs, plan_outputs)=}")

       

        # switch inference
        prediction = pt_outputs
        print(f"\n{np.min(prediction)=}, {np.max(prediction)=}\n")
        print(f"\n{prediction=}\n")

        rows = prediction.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = prediction[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
                classes_scores
            )
            if maxScore >= self.confidence_threshold:
                x1, y1, x2, y2 = prediction[0][i][0], prediction[0][i][1], prediction[0][i][2], prediction[0][i][3]
                box = [
                    int(x1), 
                    int(y1), 
                    int(x2-x1), 
                    int(y2-y1),
                ]
                # x, y, w, h = prediction[0][i][0], prediction[0][i][1], prediction[0][i][2], prediction[0][i][3]
                # box = [
                #     int(x - (0.5 * w)),
                #     int(y - (0.5 * h)),
                #     int(w),
                #     int(h),
                # ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_threshold, self.nms_threshold
            #boxes, scores, 0.5, 0.5
        )

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                #'class_name': CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            self.draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        Path(self.output_dir, "predictions").mkdir(parents=True, exist_ok=True)
        filename = str(
            Path(
                self.output_dir,
                "predictions",
                f"prediction_{rx_times[0]:.3f}_{rx_freqs[0]:.0f}Hz_{self.sample_rate:.0f}sps.png",
            )
        )
        cv2.imwrite(filename, original_image)

        return 1  # len(input_items[0])
