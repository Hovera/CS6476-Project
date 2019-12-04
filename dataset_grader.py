import glob
import os
import json


def compare_results(pred_dir, truth_dir):
    def get_pure_basename(path):
        return os.path.splitext(os.path.basename(path))[0]

    basenames = set()
    basenames.update([get_pure_basename(x) for x in
        glob.glob(os.path.join(pred_dir, '*.json'))])


    total = 0.
    correct = 0.
    for base in basenames:
        truth_path = os.path.join(truth_dir, base + '.txt')
        pred_path = os.path.join(pred_dir, base + '.json')
        if not os.path.exists(truth_path):
            continue
        if not os.path.exists(pred_path):
            continue

        predictions = process_json(pred_path)
        truths = process_truth(truth_path)

        for pred, truth in zip(predictions, truths):
            total += 1
            pred_plate, _ = pred

            if truth == pred_plate:
                correct += 1

    return correct / total, correct, total



def process_truth(filename):
    results = []
    with open(filename) as f:
        line = f.readline().strip()
        while line:
            plate = line.split('\t')[-1]
            results.append(plate)
            line = f.readline().strip()

    return results


def process_json(filename):
    with open(filename) as f:
        data = json.load(f)

    results = []
    for r in data['results']:
        plate = r['plate']
        confidence = r['confidence']
        results.append((plate, confidence))
    
    return results


if __name__ == '__main__':
    accuracy, correct, total = compare_results('dataset_ocr', 'dataset')
    print(f'Accuracy: {accuracy} ({correct} / {total})')
