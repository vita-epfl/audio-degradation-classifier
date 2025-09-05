import random
import numpy as np
import subprocess
from pathlib import Path
import torchaudio
import io

class SoxEffectGenerator:
    """
    Generates a chain of sox effects with randomized parameters.
    """
    def __init__(self, effects_config):
        """
        Initializes the generator with a configuration for effects.
        
        Args:
            effects_config (dict): A dictionary where keys are effect names
                                   and values are dicts of parameter ranges.
        """
        self.effects_config = effects_config
        self.available_effects = list(effects_config.keys())

    def generate(self, num_effects_range=(1, 5)):
        """
        Generates a random list of sox effect strings.
        
        Args:
            num_effects_range (tuple): A tuple (min, max) for the number of effects to apply.
            
        Returns:
            list: A list of sox effect strings.
        """
        num_effects = random.randint(*num_effects_range)
        effects_to_apply = random.choices(self.available_effects, k=num_effects)
        
        effect_chain = []
        for effect_name in effects_to_apply:
            params = self.effects_config[effect_name]
            param_str = f"{effect_name}"
            
            if effect_name == 'equalizer':
                # equalizer frequency width_q gain
                frequency = random.randint(*params['frequency'])
                width_q = round(random.uniform(*params['width_q']), 2)
                gain = random.randint(*params['gain'])
                param_str += f" {frequency} {width_q} {gain}"

            elif effect_name == 'reverb':
                # reverb reverberance hf_damping room_scale
                reverberance = random.randint(*params['reverberance'])
                hf_damping = random.randint(*params['hf_damping'])
                room_scale = random.randint(*params['room_scale'])
                param_str += f" {reverberance} {hf_damping} {room_scale}"
            
            elif effect_name == 'overdrive':
                gain = random.randint(*params['gain'])
                colour = random.randint(*params['colour'])
                param_str += f" {gain} {colour}"
            
            effect_chain.append(param_str)
            
        return effect_chain

    def apply_effects(self, input_waveform, sample_rate, effect_chain):
        """
        Applies effects to an in-memory audio tensor using sox.

        Args:
            input_waveform (torch.Tensor): The input audio waveform.
            sample_rate (int): The sample rate of the audio.
            effect_chain (list): The list of sox effect strings.

        Returns:
            torch.Tensor: The degraded audio waveform.
        """
        # Save input tensor to an in-memory wav file
        input_buffer = io.BytesIO()
        torchaudio.save(input_buffer, input_waveform, sample_rate, format="wav")
        input_buffer.seek(0)

        # Prepare the sox command for in-memory processing
        flat_effect_chain = []
        for effect in effect_chain:
            flat_effect_chain.extend(effect.split())
        
        command = [
            'sox', 
            '-t', 'wav', '-',  # Input from stdin
            '-t', 'wav', '-'   # Output to stdout
        ] + flat_effect_chain

        try:
            # Run sox, passing the audio via stdin and capturing stdout
            result = subprocess.run(
                command, 
                input=input_buffer.read(), 
                check=True, 
                capture_output=True
            )
            output_buffer = io.BytesIO(result.stdout)
            output_waveform, _ = torchaudio.load(output_buffer)
            return output_waveform

        except subprocess.CalledProcessError as e:
            print(f"Sox error:\n{e.stderr.decode('utf-8')}")
            raise
