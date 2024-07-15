import cv2
import numpy as np
import pickle
import streamlit as st
import time
import matplotlib.pyplot as plt

# Constants
RECT_W, RECT_H = 107, 48

def process_frame(frame):
    # Image processing pipeline
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    median_blur = cv2.medianBlur(thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(median_blur, kernel, iterations=1)
    return dilate

def check_parking_spaces(processed_frame, pos_list):
    occupied_spaces = np.zeros(len(pos_list), dtype=bool)
    for i, pos in enumerate(pos_list):
        x, y = pos
        crop = processed_frame[y:y+RECT_H, x:x+RECT_W]
        occupied_spaces[i] = cv2.countNonZero(crop) >= 900

    return occupied_spaces

def display_frame(frame, occupied_spaces, pos_list):
    free_spaces = len(pos_list) - np.count_nonzero(occupied_spaces)
    cv2.putText(frame, f'Free: {free_spaces}/{len(pos_list)}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

def display_graph(occupancy_data):
    x_values = list(occupancy_data.keys())
    y_values = list(occupancy_data.values())
    plt.plot(x_values, y_values)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Occupied Parking Spaces")
    plt.title("Parking Lot Occupancy")
    st.pyplot(plt)

def main():
    st.title("Vehicle Montioring and Insight Generation")

    uploaded_file = st.file_uploader("Choose a video file", type='mp4')

    if uploaded_file is not None:
        cap = cv2.VideoCapture(uploaded_file.name)

        with open('car_park_pos', 'rb') as f:
            pos_list = pickle.load(f)

        prev_occupied_spaces = None
        occupancy_data = {}
        frame_count = 0

        frames = []
        show_frames = False
        show_graph = False

        st.header("Analysis Options")
        st.text("Video Frames, with each update")
        show_frames_button = st.button("Show Frames")
        st.text("Visual Graph, based on video updates")
        show_graph_button = st.button("Show Graph")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame)
            occupied_spaces = check_parking_spaces(processed_frame, pos_list)

            # Update Streamlit display if there's a change
            if not np.array_equal(occupied_spaces, prev_occupied_spaces):
                frame_with_rectangles = display_frame(frame.copy(), occupied_spaces, pos_list)
                frames.append(frame_with_rectangles)
                prev_occupied_spaces = occupied_spaces.copy()

                # Update occupancy data
                occupancy_data[frame_count] = np.count_nonzero(occupied_spaces)

            frame_count += 1

            if show_frames:
                for frame in frames:
                    st.image(frame, channels="BGR")

            if show_graph:
                display_graph(occupancy_data)

            if show_frames_button:
                show_frames = True
            if show_graph_button:
                show_graph = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        st.write("Please upload a video file!")

if __name__ == "__main__":
    main()
