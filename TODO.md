# TODO

- [ ] Evaluate COAP-controlled VLA generation temperature instead of score-triggered replanning. Implemented in `configs/coap/metaworld_pool64_temperature.yaml` and queued after the fixed-threshold evaluations, including −0.5, in an isolated Brev runtime. For π0.5, the compatibility score controls the initial action-sampling noise scale from 1 to 2 while retaining the normal replanning schedule. Generation temperature is separate from the contrastive-loss temperature.
- [ ] Test whether the score-to-generation-temperature mapping can be learned.
- [ ] Investigate whether contrastive learning can be used entirely on its own for next-action-chunk prediction. Modify the observation–action alignment to target the next chunk rather than the current chunk.
