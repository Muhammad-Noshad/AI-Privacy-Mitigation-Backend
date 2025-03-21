from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
import numpy as np

def run_attack(dataset, art_classifier, attack_type):
  (x_train, y_train), (x_test, y_test) = dataset
  
  bb_attack = MembershipInferenceBlackBox(art_classifier, attack_model_type='rf')

  attack_train_ratio = 0.5
  attack_train_size = int(len(x_train) * attack_train_ratio)
  attack_test_size = int(len(x_test) * attack_train_ratio)

  bb_attack.fit(x_train[:attack_train_size], y_train[:attack_train_size], x_test[:attack_test_size], y_test[:attack_test_size])
  
  inferred_train_bb = bb_attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
  inferred_test_bb = bb_attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])

  train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
  test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))  
  attack_acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
  
  return attack_acc