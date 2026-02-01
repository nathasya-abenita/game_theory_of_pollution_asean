# Game Theory of Pollution in ASEAN
Based on my undergraduate thesis titled "Modelling Transboundary Air Pollution in ASEAN with Game Theory"

## User manual
* To install the Poetry environment, run `poetry install` in the root directory.
* In root directory, add these output folders: `output/figs`, `output/vars`

## 📝 To Do 
### Network
- [x] Add countries network scripts
- [x] Add wind speed data analysis
- [ ] Refine color palette for network plots

### Module and testing
- [x] Create module for noncooperative static game
- [x] Create module for noncooperative differential game
- [x] Create module for imitation differential game
- [ ] Create module for cooperative differential game
    - [x] Use joint optimization with `minimize` from SciPy
    - [ ] Use block coordinate descent with `minimize` from SciPy

### Simulation
- [ ] Simulate noncooperative game and finalize output
- [ ] Simulate imitation game and finalize output
- [ ] Simulate cooperative game and finalize output
- [ ] Refine paper
