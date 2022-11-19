# XRD_Tomography
This CUDA and Matlab software was used to obtain results published in

Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
by Airidas Korolkovas 
Scientific Reports volume 12, Article number: 19097 (2022)
https://doi.org/10.1038/s41598-022-23396-2

Abstract:
X-ray computed tomography (CT) is a commercially established modality for imaging large objects like passenger luggage. CT can provide the density and the effective atomic number, which is not always sufficient to identify threats like explosives and narcotics, since they can have a similar composition to benign plastics, glass, or light metals. In these cases, X-ray diffraction (XRD) may be better suited to distinguish the threats. Unfortunately, the diffracted photon flux is typically much weaker than the transmitted one. Measurement of quality XRD data is therefore slower compared to CT, which is an economic challenge for potential customers like airports. In this article we numerically analyze a novel low-cost scanner design which captures CT and XRD signals simultaneously, and uses the least possible collimation to maximize the flux. To simulate a realistic instrument, we propose a forward model that includes the resolution-limiting effects of the polychromatic spectrum, the detector, and all the finite-size geometric factors. We then show how to reconstruct XRD patterns from a large phantom with multiple diffracting objects. We include a reasonable amount of photon counting noise (Poisson statistics), as well as measurement bias (incoherent scattering). Our XRD reconstruction adds material-specific information, albeit at a low resolution, to the already existing CT image, thus improving threat detection. Our theoretical model is implemented in GPU (Graphics Processing Unit) accelerated software which can be used to further optimize scanner designs for applications in security, healthcare, and manufacturing quality control.
