import argparse
import os
import pickle


def load_original_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    if isinstance(data, dict):
        return data
    (
        videoIDs,
        videoSpeakers,
        videoLabels,
        videoSentiments,
        videoText0,
        videoText1,
        videoText2,
        videoText3,
        videoAudio,
        videoVisual,
        videoSentence,
        trainVid,
        testVid,
        _,
    ) = data
    return {
        "videoIDs": videoIDs,
        "videoSpeakers": videoSpeakers,
        "videoLabels": videoLabels,
        "videoSentiments": videoSentiments,
        "videoText0": videoText0,
        "videoText1": videoText1,
        "videoText2": videoText2,
        "videoText3": videoText3,
        "videoAudio": videoAudio,
        "videoVisual": videoVisual,
        "videoSentence": videoSentence,
        "trainVid": list(trainVid),
        "testVid": list(testVid),
    }


def build_leave_one_split(sorted_vids, valid_count):
    stride = max(1, len(sorted_vids) // valid_count)
    valid_vids = []
    selected = set()
    for idx in range(stride - 1, len(sorted_vids), stride):
        vid = sorted_vids[idx]
        valid_vids.append(vid)
        selected.add(vid)
        if len(valid_vids) == valid_count:
            break
    if len(valid_vids) < valid_count:
        for vid in sorted_vids:
            if vid in selected:
                continue
            valid_vids.append(vid)
            if len(valid_vids) == valid_count:
                break
    valid_vid_set = set(valid_vids)
    train_vids = [vid for vid in sorted_vids if vid not in valid_vid_set]
    return train_vids, valid_vids


def count_utterances(label_dict, vids):
    return int(sum(len(label_dict[vid]) for vid in vids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/meld_multi_features.pkl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/meld_multi_features_with_dev.pkl",
    )
    parser.add_argument(
        "--valid_dialogues",
        type=int,
        default=114,
    )
    args = parser.parse_args()

    dataset = load_original_dataset(args.input_path)
    original_train_vids = list(dataset["trainVid"])
    sorted_train_vids = sorted(original_train_vids)
    train_vids, dev_vids = build_leave_one_split(
        sorted_vids=sorted_train_vids,
        valid_count=args.valid_dialogues,
    )

    output_dataset = dict(dataset)
    output_dataset["trainVid"] = list(train_vids)
    output_dataset["devVid"] = list(dev_vids)
    output_dataset["testVid"] = list(dataset["testVid"])
    output_dataset["splitStrategy"] = "presplit_match_runpy_leave_one_stride"
    output_dataset["sourcePath"] = os.path.abspath(args.input_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(output_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"input_path={os.path.abspath(args.input_path)}")
    print(f"output_path={os.path.abspath(args.output_path)}")
    print(f"train_dialogues={len(train_vids)}")
    print(f"dev_dialogues={len(dev_vids)}")
    print(f"test_dialogues={len(output_dataset['testVid'])}")
    print(f"train_utterances={count_utterances(output_dataset['videoLabels'], train_vids)}")
    print(f"dev_utterances={count_utterances(output_dataset['videoLabels'], dev_vids)}")
    print(f"test_utterances={count_utterances(output_dataset['videoLabels'], output_dataset['testVid'])}")


if __name__ == "__main__":
    main()
