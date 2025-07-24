# AstroLogics

The AstroLOGIC project aims to develop a comprehensive methodology for benchmarking Boolean models, addressing a critical gap in the field of regulatory network modeling. While multiple methods exist for Boolean model synthesis, there hasn't been a standardized way to evaluate and compare these generated models.

The project proposes three main evaluation criteria:

1. Network evaluation - comparing structural similarities between models using matrix-based distance calculations
2. Logical function evaluation - analyzing and comparing the logical rules that govern node behaviors
3. Dynamic evaluation - examining state transition graphs and model behaviors through simulation

The methodology involves generating model ensembles using tools like Bonesis or BN-sketch, simulating these models using MaBoSS to identify different dynamics, and clustering models based on their behavioral similarities. This approach allows for the identification of key logical rules that produce specific dynamics and helps in understanding the relationship between model structure and behavior.

Current progress includes the development of preliminary tools for trajectory simulation and clustering, exploration of model generation from multiple sources, and visualization methods for comparing logical rules between models. The project also incorporates advanced tools for trajectory analysis and clustering, utilizing packages like traja and tslearn for comprehensive model evaluation.
