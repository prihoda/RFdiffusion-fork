inference.cyc_chains config should be a comma-separated list of _output_ chain ids that should have cyclic offsets.
Designed chains are always the first ones (in the output), so **for designing one chain that should be cyclic just use**
- `inference.cyc_chains=A`

For multiple chains, it should be possible to do 
- `inference.cyc_chains=A,B,D` however, it was not tested


implementation:
- rfdiffusion/inference/model_runners.py
    - `Sampler.get_cyclic_offset_masks()`
        - Returns masks for the RoseTTAFold model, to know with which residues use the cyclic offset. Multiple masks are possible. 
- and in the RoseTTAFold model