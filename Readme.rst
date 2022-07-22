API reference
This is the full API documentation of the imbalanced-learn toolbox.

	Under-sampling methods
		Prototype generation
			ClusterCentroids
		Prototype selection
			CondensedNearestNeighbour
			EditedNearestNeighbours
			RepeatedEditedNearestNeighbours
			AllKNN
			InstanceHardnessThreshold
			NearMiss
			NeighbourhoodCleaningRule
			OneSidedSelection
			RandomUnderSampler
			TomekLinks
	Over-sampling methods
		Basic over-sampling
			RandomOverSampler
		SMOTE algorithms
			SMOTE
			SMOTENC
			SMOTEN
			ADASYN
			BorderlineSMOTE
			KMeansSMOTE
			SVMSMOTE
	Combination of over- and under-sampling methods
		SMOTEENN
			Examples using imblearn.combine.SMOTEENN
		SMOTETomek
			Examples using imblearn.combine.SMOTETomek
	Ensemble methods
		Boosting algorithms
			EasyEnsembleClassifier
			RUSBoostClassifier
		Bagging algorithms
			BalancedBaggingClassifier
			BalancedRandomForestClassifier
	Batch generator for Keras
		BalancedBatchGenerator
			Examples using imblearn.keras.BalancedBatchGenerator
		balanced_batch_generator
	Batch generator for TensorFlow
		balanced_batch_generator
	Miscellaneous
		FunctionSampler
			Examples using imblearn.FunctionSampler
	Pipeline
		Pipeline
			Examples using imblearn.pipeline.Pipeline
		make_pipeline
			Examples using imblearn.pipeline.make_pipeline
	Metrics
		Classification metrics
			classification_report_imbalanced
			sensitivity_specificity_support
			sensitivity_score
			specificity_score
			geometric_mean_score
			macro_averaged_mean_absolute_error
			make_index_balanced_accuracy
		Pairwise metrics
			ValueDifferenceMetric
	Datasets
		make_imbalance
			Examples using imblearn.datasets.make_imbalance
		fetch_datasets
			Examples using imblearn.datasets.fetch_datasets
	Utilities
		Validation checks used in samplers
			parametrize_with_checks
			check_neighbors_object
			check_sampling_strategy
			check_target_type
		Testing compatibility of your own sampler
			parametrize_with_checks