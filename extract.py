import numpy as np


def extract_features(window_data, fps):
    right_hands = []
    left_hands = []
    frame_counts = 0

    for frame in window_data:
        frame_counts += 1
        if "hands" not in frame or len(frame["hands"]) == 0:
            continue

        for hand in frame["hands"]:
            landmarks = np.array(
                [[lm["x"], lm["y"], lm["z"]] for lm in hand["landmarks"]]
            )
            if hand["handedness"] == "Right":
                right_hands.append(landmarks)
            else:
                left_hands.append(landmarks)

    if len(right_hands) == 0 and len(left_hands) == 0:
        features = {"hand_presence_ratio": 0, "any_hand_detected": 0}
        return features

    features = {}

    features["hand_presence_ratio"] = (
        len(right_hands) + len(left_hands)
    ) / frame_counts
    features["any_hand_detected"] = 1
    features["right_hand_frames"] = len(right_hands)
    features["left_hand_frames"] = len(left_hands)

    for hand_name, hand_frames in [("right", right_hands), ("left", left_hands)]:
        if len(hand_frames) < 2:
            continue

        # shape is (num_frames, num_landmarks, 3)
        hand_data = np.array(hand_frames)

        # velocities
        if len(hand_data) >= 3:
            velocities = np.linalg.norm(hand_data[1:] - hand_data[:-1], axis=2)

            key_points = [0, 4, 8, 12]
            point_names = ["wrist", "thumb", "index", "middle"]

            for i, point_name in zip(key_points, point_names):
                if len(velocities) > 0:
                    vel_point = velocities[:, i]
                    features[f"{hand_name}_{point_name}_velocity_mean"] = np.mean(
                        vel_point
                    )
                    features[f"{hand_name}_{point_name}_velocity_std"] = np.std(
                        vel_point
                    )
                    features[f"{hand_name}_{point_name}_velocity_max"] = np.max(
                        vel_point
                    )

                    # ratio of rapid movements mean + std is threshold of rapid movement
                    vel_threshold = np.mean(vel_point) + np.std(vel_point)
                    features[f"{hand_name}_{point_name}_rapid_movement_ratio"] = np.sum(
                        vel_point > vel_threshold
                    ) / len(vel_point)

        # spatial features
        if len(hand_data) > 0:
            sample_indices = [0, -1]
            for idx, sample_idx in enumerate(sample_indices):
                if sample_idx < 0 and len(hand_data) + sample_idx < 0:
                    continue

                frame = hand_data[sample_idx]

                wrist = frame[0]
                for i, name in [(4, "thumb"), (8, "index"), (12, "middle")]:
                    dist = np.linalg.norm(frame[i] - wrist)
                    features[f"{hand_name}_wrist_to_{name}_dist_{idx}"] = dist

                fingertips = [4, 8, 12]
                fingertip_names = ["thumb", "index", "middle"]

                for i in range(len(fingertips)):
                    for j in range(i + 1, len(fingertips)):
                        pt1, pt2 = fingertips[i], fingertips[j]
                        name1, name2 = fingertip_names[i], fingertip_names[j]
                        dist = np.linalg.norm(frame[pt1] - frame[pt2])
                        features[f"{hand_name}_{name1}_to_{name2}_dist_{idx}"] = dist

        # range of motion in 3D space
        if len(hand_data) >= 2:
            for i, name in [(0, "wrist"), (4, "thumb"), (8, "index"), (12, "middle")]:
                points = hand_data[:, i, :]

                x_range = np.max(points[:, 0]) - np.min(points[:, 0])
                y_range = np.max(points[:, 1]) - np.min(points[:, 1])
                z_range = np.max(points[:, 2]) - np.min(points[:, 2])

                features[f"{hand_name}_{name}_3d_range"] = np.sqrt(
                    x_range**2 + y_range**2 + z_range**2
                )

        # angles between fingers over time
        if len(hand_data) >= 8:
            wrist_idx = 0
            thumb_indices = [1, 2, 3, 4]
            index_indices = [5, 6, 7, 8]
            middle_indices = [9, 10, 11, 12]
            ring_indices = [13, 14, 15, 16]
            pinky_indices = [17, 18, 19, 20]

            # angle time series
            angles_thumb_index = []
            angles_thumb_middle = []
            angles_index_middle = []
            angles_pinky_ring = []

            # angle at each joint
            thumb_flexion = []
            index_flexion = []
            middle_flexion = []

            # angles between fingers
            for frame in hand_data:
                wrist = frame[wrist_idx]
                thumb_tip = frame[thumb_indices[-1]]
                index_tip = frame[index_indices[-1]]
                middle_tip = frame[middle_indices[-1]]
                ring_tip = frame[ring_indices[-1]]
                pinky_tip = frame[pinky_indices[-1]]

                thumb_vec = thumb_tip - wrist
                index_vec = index_tip - wrist
                middle_vec = middle_tip - wrist
                ring_vec = ring_tip - wrist
                pinky_vec = pinky_tip - wrist

                thumb_vec_norm = thumb_vec / np.linalg.norm(thumb_vec)
                index_vec_norm = index_vec / np.linalg.norm(index_vec)
                middle_vec_norm = middle_vec / np.linalg.norm(middle_vec)
                ring_vec_norm = ring_vec / np.linalg.norm(ring_vec)
                pinky_vec_norm = pinky_vec / np.linalg.norm(pinky_vec)

                cos_thumb_index = np.dot(thumb_vec_norm, index_vec_norm)
                angle_thumb_index = (
                    np.arccos(np.clip(cos_thumb_index, -1.0, 1.0)) * 180 / np.pi
                )
                angles_thumb_index.append(angle_thumb_index)

                cos_thumb_middle = np.dot(thumb_vec_norm, middle_vec_norm)
                angle_thumb_middle = (
                    np.arccos(np.clip(cos_thumb_middle, -1.0, 1.0)) * 180 / np.pi
                )
                angles_thumb_middle.append(angle_thumb_middle)

                cos_index_middle = np.dot(index_vec_norm, middle_vec_norm)
                angle_index_middle = (
                    np.arccos(np.clip(cos_index_middle, -1.0, 1.0)) * 180 / np.pi
                )
                angles_index_middle.append(angle_index_middle)

                cos_pinky_ring = np.dot(pinky_vec_norm, ring_vec_norm)
                angle_pinky_ring = (
                    np.arccos(np.clip(cos_pinky_ring, -1.0, 1.0)) * 180 / np.pi
                )
                angles_pinky_ring.append(angle_pinky_ring)

                # flexion angles for thumb, index, and middle fingers
                thumb_mcp = frame[thumb_indices[1]]
                thumb_ip = frame[thumb_indices[2]]
                thumb_tip = frame[thumb_indices[3]]
                vec1 = thumb_ip - thumb_mcp
                vec2 = thumb_tip - thumb_ip
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    vec1 = vec1 / np.linalg.norm(vec1)
                    vec2 = vec2 / np.linalg.norm(vec2)
                    cos_angle = np.dot(vec1, vec2)
                    flexion = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                    thumb_flexion.append(flexion)
                else:
                    thumb_flexion.append(0)

                index_mcp = frame[index_indices[0]]
                index_pip = frame[index_indices[1]]
                index_dip = frame[index_indices[2]]
                vec1 = index_pip - index_mcp
                vec2 = index_dip - index_pip
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    vec1 = vec1 / np.linalg.norm(vec1)
                    vec2 = vec2 / np.linalg.norm(vec2)
                    cos_angle = np.dot(vec1, vec2)
                    flexion = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                    index_flexion.append(flexion)
                else:
                    index_flexion.append(0)

                middle_mcp = frame[middle_indices[0]]
                middle_pip = frame[middle_indices[1]]
                middle_dip = frame[middle_indices[2]]
                vec1 = middle_pip - middle_mcp
                vec2 = middle_dip - middle_pip
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    vec1 = vec1 / np.linalg.norm(vec1)
                    vec2 = vec2 / np.linalg.norm(vec2)
                    cos_angle = np.dot(vec1, vec2)
                    flexion = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                    middle_flexion.append(flexion)
                else:
                    middle_flexion.append(0)

            angles_thumb_index = np.array(angles_thumb_index)
            angles_thumb_middle = np.array(angles_thumb_middle)
            angles_index_middle = np.array(angles_index_middle)
            angles_pinky_ring = np.array(angles_pinky_ring)
            thumb_flexion = np.array(thumb_flexion)
            index_flexion = np.array(index_flexion)
            middle_flexion = np.array(middle_flexion)

            # fft features
            if len(angles_thumb_index) >= 8:
                # windowing to reduce spectral leakage
                window = np.hamming(len(angles_thumb_index))

                fft_thumb_index = np.abs(np.fft.rfft(angles_thumb_index * window))
                fft_thumb_middle = np.abs(np.fft.rfft(angles_thumb_middle * window))
                fft_index_middle = np.abs(np.fft.rfft(angles_index_middle * window))
                fft_pinky_ring = np.abs(np.fft.rfft(angles_pinky_ring * window))
                fft_thumb_flexion = np.abs(np.fft.rfft(thumb_flexion * window))
                fft_index_flexion = np.abs(np.fft.rfft(index_flexion * window))
                fft_middle_flexion = np.abs(np.fft.rfft(middle_flexion * window))

                freq_bins = np.fft.rfftfreq(len(angles_thumb_index), d=1 / fps)

                # dominant frequency
                if len(freq_bins) > 0 and len(fft_thumb_index) > 0:
                    features[f"{hand_name}_thumb_index_dom_freq"] = freq_bins[
                        np.argmax(fft_thumb_index)
                    ]
                    features[f"{hand_name}_thumb_middle_dom_freq"] = freq_bins[
                        np.argmax(fft_thumb_middle)
                    ]
                    features[f"{hand_name}_index_middle_dom_freq"] = freq_bins[
                        np.argmax(fft_index_middle)
                    ]
                    features[f"{hand_name}_pinky_ring_dom_freq"] = freq_bins[
                        np.argmax(fft_pinky_ring)
                    ]
                    features[f"{hand_name}_thumb_flexion_dom_freq"] = freq_bins[
                        np.argmax(fft_thumb_flexion)
                    ]
                    features[f"{hand_name}_index_flexion_dom_freq"] = freq_bins[
                        np.argmax(fft_index_flexion)
                    ]
                    features[f"{hand_name}_middle_flexion_dom_freq"] = freq_bins[
                        np.argmax(fft_middle_flexion)
                    ]

                    # low frequency band
                    low_freq_idx = np.where(freq_bins <= 2)[0]
                    if len(low_freq_idx) > 0:
                        features[f"{hand_name}_thumb_index_low_freq_energy"] = np.sum(
                            fft_thumb_index[low_freq_idx]
                        ) / np.sum(fft_thumb_index)
                        features[f"{hand_name}_thumb_middle_low_freq_energy"] = np.sum(
                            fft_thumb_middle[low_freq_idx]
                        ) / np.sum(fft_thumb_middle)
                        features[f"{hand_name}_index_middle_low_freq_energy"] = np.sum(
                            fft_index_middle[low_freq_idx]
                        ) / np.sum(fft_index_middle)

                    # mid frequency band
                    mid_freq_idx = np.where((freq_bins > 2) & (freq_bins <= 5))[0]
                    if len(mid_freq_idx) > 0:
                        features[f"{hand_name}_thumb_index_mid_freq_energy"] = np.sum(
                            fft_thumb_index[mid_freq_idx]
                        ) / np.sum(fft_thumb_index)
                        features[f"{hand_name}_thumb_middle_mid_freq_energy"] = np.sum(
                            fft_thumb_middle[mid_freq_idx]
                        ) / np.sum(fft_thumb_middle)
                        features[f"{hand_name}_index_middle_mid_freq_energy"] = np.sum(
                            fft_index_middle[mid_freq_idx]
                        ) / np.sum(fft_index_middle)

                    # high frequency band
                    high_freq_idx = np.where(freq_bins > 5)[0]
                    if len(high_freq_idx) > 0:
                        features[f"{hand_name}_thumb_index_high_freq_energy"] = np.sum(
                            fft_thumb_index[high_freq_idx]
                        ) / np.sum(fft_thumb_index)
                        features[f"{hand_name}_thumb_middle_high_freq_energy"] = np.sum(
                            fft_thumb_middle[high_freq_idx]
                        ) / np.sum(fft_thumb_middle)
                        features[f"{hand_name}_index_middle_high_freq_energy"] = np.sum(
                            fft_index_middle[high_freq_idx]
                        ) / np.sum(fft_index_middle)

                # spectral centroid (weigheted by frequency bins)
                if len(freq_bins) > 0:
                    features[f"{hand_name}_thumb_index_spectral_centroid"] = np.sum(
                        freq_bins * fft_thumb_index
                    ) / np.sum(fft_thumb_index)
                    features[f"{hand_name}_thumb_middle_spectral_centroid"] = np.sum(
                        freq_bins * fft_thumb_middle
                    ) / np.sum(fft_thumb_middle)
                    features[f"{hand_name}_index_middle_spectral_centroid"] = np.sum(
                        freq_bins * fft_index_middle
                    ) / np.sum(fft_index_middle)
                    features[f"{hand_name}_flexion_spectral_centroid"] = np.sum(
                        freq_bins
                        * (fft_thumb_flexion + fft_index_flexion + fft_middle_flexion)
                    ) / np.sum(
                        fft_thumb_flexion + fft_index_flexion + fft_middle_flexion
                    )

            # more stats
            features[f"{hand_name}_thumb_index_angle_mean"] = np.mean(
                angles_thumb_index
            )
            features[f"{hand_name}_thumb_index_angle_std"] = np.std(angles_thumb_index)
            features[f"{hand_name}_thumb_index_angle_min"] = np.min(angles_thumb_index)
            features[f"{hand_name}_thumb_index_angle_max"] = np.max(angles_thumb_index)
            features[f"{hand_name}_thumb_index_angle_range"] = np.max(
                angles_thumb_index
            ) - np.min(angles_thumb_index)

            features[f"{hand_name}_thumb_middle_angle_mean"] = np.mean(
                angles_thumb_middle
            )
            features[f"{hand_name}_thumb_middle_angle_std"] = np.std(
                angles_thumb_middle
            )
            features[f"{hand_name}_thumb_middle_angle_range"] = np.max(
                angles_thumb_middle
            ) - np.min(angles_thumb_middle)

            features[f"{hand_name}_index_middle_angle_mean"] = np.mean(
                angles_index_middle
            )
            features[f"{hand_name}_index_middle_angle_std"] = np.std(
                angles_index_middle
            )
            features[f"{hand_name}_index_middle_angle_range"] = np.max(
                angles_index_middle
            ) - np.min(angles_index_middle)

            features[f"{hand_name}_thumb_flexion_mean"] = np.mean(thumb_flexion)
            features[f"{hand_name}_thumb_flexion_std"] = np.std(thumb_flexion)
            features[f"{hand_name}_thumb_flexion_range"] = np.max(
                thumb_flexion
            ) - np.min(thumb_flexion)

            features[f"{hand_name}_index_flexion_mean"] = np.mean(index_flexion)
            features[f"{hand_name}_index_flexion_std"] = np.std(index_flexion)
            features[f"{hand_name}_index_flexion_range"] = np.max(
                index_flexion
            ) - np.min(index_flexion)

            features[f"{hand_name}_middle_flexion_mean"] = np.mean(middle_flexion)
            features[f"{hand_name}_middle_flexion_std"] = np.std(middle_flexion)
            features[f"{hand_name}_middle_flexion_range"] = np.max(
                middle_flexion
            ) - np.min(middle_flexion)

            features[f"{hand_name}_thumb_index_middle_corr"] = np.corrcoef(
                angles_thumb_index, angles_index_middle
            )[0, 1]
            features[f"{hand_name}_flexion_corr"] = np.corrcoef(
                thumb_flexion, index_flexion
            )[0, 1]

        # hand shape stability (variance of distances between fingertips)
        if len(hand_data) >= 5:
            stability_features = {}

            all_fingers = [4, 8, 12, 16, 20]
            finger_names = ["thumb", "index", "middle", "ring", "pinky"]

            for i in range(len(all_fingers)):
                for j in range(i + 1, len(all_fingers)):
                    pt1, pt2 = all_fingers[i], all_fingers[j]
                    name1, name2 = finger_names[i], finger_names[j]

                    distances = []
                    for frame in hand_data:
                        dist = np.linalg.norm(frame[pt1] - frame[pt2])
                        distances.append(dist)

                    if len(distances) > 0:
                        # stability is inverse of variation coefficient
                        mean_dist = np.mean(distances)
                        std_dist = np.std(distances)
                        if mean_dist > 0:
                            stability = 1.0 - (std_dist / mean_dist)
                            stability_features[
                                f"{hand_name}_{name1}_{name2}_stability"
                            ] = stability

            if stability_features:
                features[f"{hand_name}_overall_shape_stability"] = np.mean(
                    list(stability_features.values())
                )
                features.update(stability_features)

        # trajectory features
        if len(hand_data) >= 8:
            trajectory_points = [0, 4, 8, 12]
            point_names = ["wrist", "thumb", "index", "middle"]

            for idx, point_name in zip(trajectory_points, point_names):
                trajectory = hand_data[:, idx, :]

                if len(trajectory) >= 8:
                    # distance travelled
                    displacements = np.linalg.norm(
                        trajectory[1:] - trajectory[:-1], axis=1
                    )
                    total_distance = np.sum(displacements)
                    features[f"{hand_name}_{point_name}_total_distance"] = (
                        total_distance
                    )

                    # how straight is the path?
                    direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
                    if total_distance > 0:
                        straightness = direct_distance / total_distance
                        features[f"{hand_name}_{point_name}_path_straightness"] = (
                            straightness
                        )

                    for coord_idx, coord_name in enumerate(["x", "y", "z"]):
                        trajectory_coord = trajectory[:, coord_idx]

                        window = np.hamming(len(trajectory_coord))
                        fft_result = np.abs(np.fft.rfft(trajectory_coord * window))

                        if len(fft_result) > 0:
                            norm_fft = fft_result / np.sum(fft_result)
                            entropy = -np.sum(norm_fft * np.log2(norm_fft + 1e-10))
                            features[
                                f"{hand_name}_{point_name}_{coord_name}_spectral_entropy"
                            ] = entropy

                            freq_bins = np.fft.rfftfreq(
                                len(trajectory_coord), d=1 / fps
                            )
                            if len(freq_bins) > 0:
                                dom_freq_idx = np.argmax(fft_result)
                                features[
                                    f"{hand_name}_{point_name}_{coord_name}_dom_freq"
                                ] = freq_bins[dom_freq_idx]

                                low_freq_idx = np.where(freq_bins <= 2)[0]
                                if len(low_freq_idx) > 0:
                                    features[
                                        f"{hand_name}_{point_name}_{coord_name}_low_freq_energy"
                                    ] = np.sum(fft_result[low_freq_idx]) / np.sum(
                                        fft_result
                                    )

    return features
