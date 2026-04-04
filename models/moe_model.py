import os
import numpy as np
import tensorflow as tf

class HierarchicalMoE:
    def __init__(self, models_dir='saved_models'):
        """
        Loads all the trained experts and gating networks to form the full model.
        """
        self.models_dir = models_dir
        
        # Load Experts
        print("Loading experts...")
        self.base_expert = tf.keras.models.load_model(os.path.join(models_dir, 'base_expert_final.keras'))
        self.art_expert = tf.keras.models.load_model(os.path.join(models_dir, 'artificial_expert_final.keras'))
        self.nat_expert = tf.keras.models.load_model(os.path.join(models_dir, 'natural_expert_final.keras'))
        
        # Load Gating Networks
        print("Loading gating networks...")
        self.first_level_gate = tf.keras.models.load_model(os.path.join(models_dir, 'first_level_gate_final.keras'))
        self.art_gater = tf.keras.models.load_model(os.path.join(models_dir, 'artificial_gater_final.keras'))
        self.nat_gater = tf.keras.models.load_model(os.path.join(models_dir, 'natural_gater_final.keras'))
        
        print("All models loaded successfully.")
        
    def predict(self, x, batch_size=32, verbose=1):
        """
        Performs inference using the Hierarchical MoE routing logic.
        """
        # 1. First-level routing probabilities (0: Artificial, 1: Natural)
        p_nat = self.first_level_gate.predict(x, batch_size=batch_size, verbose=verbose)
        p_art = 1.0 - p_nat
        
        # 2. Second-level routing weights [w_base, w_spec]
        w_art = self.art_gater.predict(x, batch_size=batch_size, verbose=verbose)
        w_nat = self.nat_gater.predict(x, batch_size=batch_size, verbose=verbose)
        
        # 3. Expert predictions
        pred_base = self.base_expert.predict(x, batch_size=batch_size, verbose=verbose)
        pred_art = self.art_expert.predict(x, batch_size=batch_size, verbose=verbose)
        pred_nat = self.nat_expert.predict(x, batch_size=batch_size, verbose=verbose)
        
        # 4. Integrate predictions
        # Artificial Branch
        w_base_art = w_art[:, 0:1]
        w_spec_art = w_art[:, 1:2]
        branch_art = w_base_art * pred_base + w_spec_art * pred_art
        
        # Natural Branch
        w_base_nat = w_nat[:, 0:1]
        w_spec_nat = w_nat[:, 1:2]
        branch_nat = w_base_nat * pred_base + w_spec_nat * pred_nat
        
        # Final Hierarchical Combination
        final_preds = p_art * branch_art + p_nat * branch_nat
        
        return final_preds
        
    def evaluate(self, x, y_true_oh, batch_size=32):
        from tensorflow.keras.metrics import CategoricalAccuracy
        preds = self.predict(x, batch_size=batch_size, verbose=0)
        acc = CategoricalAccuracy()
        acc.update_state(y_true_oh, preds)
        return acc.result().numpy()
        
    def get_routing_info(self, x):
        """
        Returns routing probabilities and expert selections for visualization.
        """
        p_nat = self.first_level_gate.predict(x, verbose=0)
        w_art = self.art_gater.predict(x, verbose=0)
        w_nat = self.nat_gater.predict(x, verbose=0)
        
        return {
            'prob_natural': p_nat,
            'prob_artificial': 1.0 - p_nat,
            'weight_art_branch': w_art,
            'weight_nat_branch': w_nat
        }
