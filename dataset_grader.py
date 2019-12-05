import glob
import os
import json
import sys

from collections import defaultdict


def compare_results_k(pred_dir, truth_dir):
    def get_pure_basename(path):
        # return os.path.splitext(os.path.basename(path))[0]
        return os.path.basename(path).split('.')[0]

    basenames = set()
    basenames.update([get_pure_basename(x) for x in
        glob.glob(os.path.join(truth_dir, '*.txt'))])


    total = 0.
    correct = 0.
    results = []
    for base in basenames:
        truth_path = os.path.join(truth_dir, base + '.txt')
        pred_paths = [os.path.join(pred_dir, base + f'_center_{x}.json') for x
                in range(3)]
        if not os.path.exists(truth_path):
            continue

        pred_exist = [not os.path.exists(p) for p in pred_paths]
        if sum(pred_exist) > 0:
            continue

        predictions = set()
        for pred_path in pred_paths:
            predictions.update(process_json(pred_path))
        truths = process_truth(truth_path)
        
        total += len(predictions.union(truths))
        for plate in truths:
            if plate in predictions:
                correct += 1
                results.append((plate, "correct"))
            else:
                results.append((plate, "incorrect"))

    with open('results.txt', 'w') as f:
        for r in results:
            f.write(f'{r}\n')

    if total > 0:
        return correct / total, correct, total
    else:
        return 0, 0, 0


def compare_results(pred_dir, truth_dir):
    def get_pure_basename(path):
        # return os.path.splitext(os.path.basename(path))[0]
        return os.path.basename(path).split('.')[0]

    basenames = set()
    basenames.update([get_pure_basename(x) for x in
        glob.glob(os.path.join(truth_dir, '*.txt'))])


    total = 0.
    correct = 0.
    results = []
    for base in basenames:
        truth_path = os.path.join(truth_dir, base + '.txt')
        pred_path = os.path.join(pred_dir, base + '.json')
        if not os.path.exists(truth_path):
            continue

        if not os.path.exists(pred_path):
            continue

        predictions = process_json(pred_path)
        truths = process_truth(truth_path)
        
        for plate in truths:
            total += 1
            if plate in predictions:
                correct += 1
                results.append((plate, "correct"))
            else:
                results.append((plate, "incorrect"))

    with open('results.txt', 'w') as f:
        for r in results:
            f.write(f'{r}\n')

    if total > 0:
        return correct / total, correct, total
    else:
        return 0, 0, 0

def process_truth(filename):
    results = []
    with open(filename) as f:
        line = f.readline().strip()
        while line:
            plate = line.split('\t')[-1]
            results.append(plate)
            line = f.readline().strip()

    return set(results)


def process_json(filename):
    with open(filename) as f:
        data = json.load(f)

    results = []
    for r in data['results']:
        plate = r['plate']
        # confidence = r['confidence']
        results.append(plate)
    
    return set(results)


def see(pred_dir):
    res = defaultdict(list)
    for path in glob.glob(os.path.join(pred_dir, '*.json')):
        results = process_json(path)
        res[path] = results

    with open('output.txt', 'w') as f:
        for path, result in res.items():
            f.write(f'{path}: {result}\n')


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        _, pred_dir, truth_dir, *_ = sys.argv
        accuracy, correct, total = compare_results_k(pred_dir, truth_dir)
    else:
        accuracy, correct, total = compare_results('data/dataset_ocr',
                'data/dataset')
    # see('data/kmeans_ocr')
    print(f'Accuracy: {accuracy} ({correct} / {total})')
