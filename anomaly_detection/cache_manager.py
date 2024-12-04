import os
import hashlib
import json
import pickle

class CacheManager:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_cache(self, step_name, config):
        """
        Retrieves cached data if configuration matches.

        Parameters:
            step_name (str): Name of the step (e.g., 'step1')
            config (dict): Configuration dictionary for the step

        Returns:
            data: Cached data if available and config matches, else None
        """
        config_hash = self._get_config_hash(config)
        cache_file = os.path.join(self.cache_dir, f'{step_name}_cache.pkl')
        config_file = os.path.join(self.cache_dir, f'{step_name}_config.txt')

        if os.path.exists(cache_file) and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                cached_config_hash = f.read()
            if cached_config_hash == config_hash:
                print(f"Loading cached data for {step_name}...")
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"Cached data for {step_name} loaded.")
                return data
            else:
                print(f"Configurations have changed for {step_name}. Re-running {step_name}.")
                return None
        else:
            print(f"No cache found for {step_name}. Running {step_name}.")
            return None

    def save_cache(self, step_name, config, data):
        """
        Saves data and configuration hash to cache.

        Parameters:
            step_name (str): Name of the step (e.g., 'step1')
            config (dict): Configuration dictionary for the step
            data: Data to cache
        """
        config_hash = self._get_config_hash(config)
        cache_file = os.path.join(self.cache_dir, f'{step_name}_cache.pkl')
        config_file = os.path.join(self.cache_dir, f'{step_name}_config.txt')

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        with open(config_file, 'w') as f:
            f.write(config_hash)

    def _get_config_hash(self, config):
        """
        Generates a SHA-256 hash of the configuration dictionary.

        Parameters:
            config (dict): Configuration dictionary

        Returns:
            str: SHA-256 hash of the configuration
        """
        config_json = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode('utf-8')).hexdigest()
        return config_hash
