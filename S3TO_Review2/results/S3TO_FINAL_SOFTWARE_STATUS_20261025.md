# S3TO — Software Validation and Results Handoff

## Validation status

- Python syntax checks passed: 42 files.
- Automated regression suite passed; see terminal output from this packaging run.
- Path A completed three Mininet seeds (42, 43, 44), with 10 rounds per mode.
- Path B candidate-1 assignment completed a 10-round Mininet run.
- Physical Raspberry Pi validation has not yet been performed in this software environment.

## Path A results

| Seed | Final S3TO and direct accuracy | S3TO global-host upload bytes | S3TO byte-hop proxy | S3TO mean round seconds | Direct mean round seconds |
|---:|---:|---:|---:|---:|---:|
| 42 | 41.51% | 11,720,900 | 70,609,484 | 20.845 | 13.976 |
| 43 | 41.13% | 11,720,900 | 70,611,266 | 20.654 | 14.516 |
| 44 | 39.27% | 11,721,890 | 70,612,718 | 20.613 | 15.005 |

Mean final accuracy across the three seeds: 40.64%.
Sample standard deviation: 1.20 percentage points.

The saved comparison reports equal final accuracy for hierarchical S3TO and direct FedAvg within each seed. S3TO reduced the measured global-host upload traffic by about 66.67%, but its forwarding byte-hop proxy and runtime were higher.

## Path B: candidate-1 grouping vs original S3TO grouping

- Original assignment: {'0': 'edge1', '1': 'edge1', '3': 'edge1', '2': 'edge2', '4': 'edge2', '5': 'edge2'}
- Candidate-1 assignment: {'0': 'edge1', '1': 'edge1', '2': 'edge1', '3': 'edge2', '4': 'edge2', '5': 'edge2'}
- Both runs used the same seed-42 configuration and completed 10 rounds.
- All round accuracies identical: True
- Original final accuracy: 41.51%.
- Alternative final accuracy: 41.51%.
- Mean runtime difference, alternative minus original: +0.0270 seconds/round.
- Global-host upload difference: -1,584 bytes.
- Forwarding byte-hop proxy difference: +4,620 bytes.

These differences do not establish a meaningful performance advantage for this alternative grouping.

## Important limitations

1. The Mininet runner trains client models in the runner process and transports the updates through the emulated network. It does not broadcast global model weights back to each emulated client namespace over Mininet. The network counters are not complete FL round-trip totals.
2. The byte-hop proxy counts bytes on multiple switch hops; it is not unique end-to-end traffic or payload-only volume.
3. The Path B metric scripts evaluate label-distribution objectives. A complete controlled model/network performance ablation across all grouping candidates remains future work.
4. Physical Raspberry Pi training, aggregation, and network measurements are not yet verified.
5. The controller/network experiment is intended for a trusted, isolated lab network; the patched protocol does not provide TLS or cryptographic client authentication.

## Key artifacts

- Path A comparison JSON files for seeds 42, 43 and 44.
- Path A hierarchical/direct raw result files.
- Path B candidate-1 result file.
- Controller, aggregation, networking, model, data, grouping and physical-client source files.
- Configurations, tests, and project notes.
