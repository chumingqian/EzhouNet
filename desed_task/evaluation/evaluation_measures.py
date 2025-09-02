import os

import numpy as np
import pandas as pd
import psds_eval
import sed_eval
import sed_scores_eval
from psds_eval import PSDSEval, plot_psd_roc

from sed_eval.sound_event import  SoundEventMetrics
def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    Args:
        df: pd.DataFrame, the dataframe to search on
        fname: the filename to extract the value from the dataframe
    Returns:
         list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict("records")
    else:
        event_list_for_current_file = event_file.to_dict("records")

    return event_list_for_current_file


def psds_results(psds_obj):
    """Compute psds scores
    Args:
        psds_obj: psds_eval.PSDSEval object with operating points.
    Returns:
    """
    try:
        psds_score = psds_obj.psds(alpha_ct=0, alpha_st=0, max_efpr=100)
        print(f"\nPSD-Score (0, 0, 100): {psds_score.value:.5f}")
        psds_score = psds_obj.psds(alpha_ct=1, alpha_st=0, max_efpr=100)
        print(f"\nPSD-Score (1, 0, 100): {psds_score.value:.5f}")
        psds_score = psds_obj.psds(alpha_ct=0, alpha_st=1, max_efpr=100)
        print(f"\nPSD-Score (0, 1, 100): {psds_score.value:.5f}")
    except psds_eval.psds.PSDSEvalError as e:
        print("psds did not work ....")
        raise EnvironmentError


def event_based_evaluation_df(
    reference, estimated, t_collar=0.200, percentage_of_length=0.2
):
    """Calculate EventBasedMetric given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
         sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric

import   dcase_util
def event_based_evaluation_df_union(
    reference, estimated, t_collar=0.200, percentage_of_length=0.2
):
    """Calculate EventBasedMetric given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename", "onset", "offset", and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename", "onset", "offset", and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
        EventBasedMetrics instance with the scores
    """

    # Get all unique filenames from both reference and estimated dataframes
    reference_files = reference["filename"].unique()
    estimated_files = estimated["filename"].unique()
    all_files = set(reference_files) | set(estimated_files)

    # Collect all unique event labels from both reference and estimated dataframes
    classes = set(reference["event_label"].dropna().unique()).union(
        set(estimated["event_label"].dropna().unique())
    )
    classes = list(classes)

    # Note, Initialize the Update_EventBasedMetrics class
    event_based_metric = Update_EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )

    # Convert the reference and estimated dataframes into MetaDataContainers
    reference_event_list = dcase_util.containers.MetaDataContainer()
    for _, row in reference.iterrows():
        event = {
            "filename": row["filename"],
            "onset": row["onset"],
            "offset": row["offset"],
            "event_label": row["event_label"],
        }
        reference_event_list.append(event)

    estimated_event_list = dcase_util.containers.MetaDataContainer()
    for _, row in estimated.iterrows():
        event = {
            "filename": row["filename"],
            "onset": row["onset"],
            "offset": row["offset"],
            "event_label": row["event_label"],
        }
        estimated_event_list.append(event)

    # Evaluate per file
    for filename in all_files:
        # Get per-file reference and estimated events
        ref_events = reference_event_list.filter(filename=filename)
        est_events = estimated_event_list.filter(filename=filename)

        # Initialize empty event lists if necessary
        if ref_events is None:
            ref_events = dcase_util.containers.MetaDataContainer()
        if est_events is None:
            est_events = dcase_util.containers.MetaDataContainer()

        # Evaluate the events for this file
        event_based_metric.evaluate_file(ref_events, est_events)

    return event_based_metric





def segment_based_evaluation_df(reference, estimated, time_resolution=1.0):
    """Calculate SegmentBasedMetrics given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        time_resolution: float, the time resolution of the segment based metric
    Returns:
         sed_eval.sound_event.SegmentBasedMetrics with the scores
    """
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return segment_based_metric










def compute_sed_eval_metrics(predictions, groundtruth):
    """Compute sed_eval metrics event based and segment based with default parameters used in the task.
    Args:
        predictions: pd.DataFrame, predictions dataframe
        groundtruth: pd.DataFrame, groundtruth dataframe
    Returns:
        tuple, (sed_eval.sound_event.EventBasedMetrics, sed_eval.sound_event.SegmentBasedMetrics)
    """
    # metric_event = event_based_evaluation_df(
    #     groundtruth, predictions, t_collar=0.200, percentage_of_length= 0.1 #0.2
    # )

    metric_event = event_based_evaluation_df_union(
        groundtruth, predictions, t_collar=0.200, percentage_of_length= 0.1 #0.2
    )


    metric_segment = segment_based_evaluation_df(
        groundtruth, predictions, time_resolution= 0.2  #1.0
    )

    return metric_event, metric_segment


def compute_per_intersection_macro_f1(
    prediction_dfs,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
):
    """Compute F1-score per intersection, using the defautl
    Args:
        prediction_dfs: dict, a dictionary with thresholds keys and predictions dataframe
        ground_truth_file: pd.DataFrame, the groundtruth dataframe
        durations_file: pd.DataFrame, the duration dataframe
        dtc_threshold: float, the parameter used in PSDSEval, percentage of tolerance for groundtruth intersection
            with predictions
        gtc_threshold: float, the parameter used in PSDSEval percentage of tolerance for predictions intersection
            with groundtruth
        gtc_threshold: float, the parameter used in PSDSEval to know the percentage needed to count FP as cross-trigger

    Returns:

    """
    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")

    psds = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )
    psds_macro_f1 = []
    for threshold in prediction_dfs.keys():
        if not prediction_dfs[threshold].empty:
            threshold_f1, _ = psds.compute_macro_f_score(prediction_dfs[threshold])
        else:
            threshold_f1 = 0
        if np.isnan(threshold_f1):
            threshold_f1 = 0.0
        psds_macro_f1.append(threshold_f1)
    psds_macro_f1 = np.mean(psds_macro_f1)
    return psds_macro_f1


def compute_psds_from_operating_points(
    prediction_dfs,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    save_dir=None,
):
    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")
    psds_eval = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        # see issue https://github.com/audioanalytic/psds_eval/issues/3
        det["index"] = range(1, len(det) + 1)
        det = det.set_index("index")
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        pred_dir = os.path.join(
            save_dir,
            f"predictions_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}",
        )
        os.makedirs(pred_dir, exist_ok=True)
        for k in prediction_dfs.keys():
            prediction_dfs[k].to_csv(
                os.path.join(pred_dir, f"predictions_th_{k:.2f}.tsv"),
                sep="\t",
                index=False,
            )

        filename = (
            f"PSDS_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}"
            f"_ct{alpha_ct}_st{alpha_st}_max{max_efpr}_psds_eval.png"
        )
        plot_psd_roc(
            psds_score,
            filename=os.path.join(save_dir, filename),
        )

    return psds_score.value


def compute_psds_from_scores(
    scores,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    num_jobs= 1,  # set 0 for debug 4,
    save_dir=None,
):
    psds, single_class_psds, psd_roc, single_class_rocs, *_ = (
        sed_scores_eval.intersection_based.psds(
            scores=scores,
            ground_truth=ground_truth_file,
            audio_durations=durations_file,
            dtc_threshold=dtc_threshold,
            gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            alpha_ct=alpha_ct,
            alpha_st=alpha_st,
            max_efpr=max_efpr,
            num_jobs=num_jobs,
        )
    )
    if save_dir is not None:
        scores_dir = os.path.join(save_dir, "scores")
        sed_scores_eval.io.write_sed_scores(scores, scores_dir)
        filename = (
            f"PSDS_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}"
            f"_ct{alpha_ct}_st{alpha_st}_max{max_efpr}_sed_scores_eval.png"
        )
        sed_scores_eval.utils.visualization.plot_psd_roc(
            psd_roc,
            filename=os.path.join(save_dir, filename),
            dtc_threshold=dtc_threshold,
            gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            alpha_ct=alpha_ct,
            alpha_st=alpha_st,
            unit_of_time="hour",
            max_efpr=max_efpr,
            psds=psds,
        )
    return psds



import numpy
import sed_eval.metric as metric
import sed_eval.util  as utils
# import math
# import dcase_util
from dcase_util import containers

class Update_EventBasedMetrics(SoundEventMetrics):
    def __init__(self,
                 event_label_list,
                 evaluate_onset=True,
                 evaluate_offset=True,
                 t_collar=0.200,
                 percentage_of_length=0.5,
                 event_matching_type='optimal',
                 empty_system_output_handling='zero_score',
                 **kwargs):
        """Constructor

        Parameters
        ----------
        event_label_list : list
            List of unique event labels

        evaluate_onset : bool
            Evaluate onset.
            Default value True

        evaluate_offset : bool
            Evaluate offset.
            Default value True

        t_collar : float (0,]
            Time collar used when evaluating validity of the onset and offset, in seconds.
            Default value 0.2

        percentage_of_length : float in [0, 1]
            Second condition, percentage of the length within which the estimated offset has to be in order to be
            considered a valid estimation.
            Default value 0.5

        event_matching_type : str
            Event matching type. Set 'optimal' for graph-based matching, or 'greedy' for always select first found match.
            Default value 'optimal'

        empty_system_output_handling : str
            How to handle empty system outputs ('zero_score' or 'non_zero_score').
            Default value 'zero_score'

        """
        SoundEventMetrics.__init__(self, **kwargs)

        if isinstance(event_label_list, numpy.ndarray) and len(event_label_list.shape) == 1:
            # We have numpy array, convert it to list
            event_label_list = event_label_list.tolist()

        if not isinstance(event_label_list, list):
            raise ValueError(
                "event_label_list needs to be list or numpy.array"
            )

        if not isinstance(t_collar, float) or t_collar <= 0.0:
            raise ValueError(
                "t_collar needs to be float > 0"
            )

        if not isinstance(percentage_of_length, float) or percentage_of_length < 0.0 or percentage_of_length > 1.0:
            raise ValueError(
                "t_collar percentage_of_length to be float in [0, 1]"
            )


        # Initialize parameters
        self.event_label_list = event_label_list
        self.evaluate_onset = evaluate_onset
        self.evaluate_offset = evaluate_offset
        self.t_collar = t_collar
        self.percentage_of_length = percentage_of_length
        self.event_matching_type = event_matching_type
        self.empty_system_output_handling = empty_system_output_handling

        self.evaluated_length = 0.0
        self.evaluated_files = 0

        if not evaluate_onset and not evaluate_offset:
            raise ValueError("Both evaluate_onset and evaluate_offset cannot be set to False")

        # Initialize overall metrics
        self.overall = {
            'Nref': 0.0,
            'Nsys': 0.0,
            'Nsubs': 0.0,
            'Ntp': 0.0,
            'Nfp': 0.0,
            'Nfn': 0.0,
        }

        # Initialize class-wise metrics
        self.class_wise = {}
        for class_label in self.event_label_list:
            self.class_wise[class_label] = {
                'Nref': 0.0,
                'Nsys': 0.0,
                'Ntp': 0.0,
                'Nfp': 0.0,
                'Nfn': 0.0,
            }


    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self.results()

    def __str__(self):
        """Print result reports"""

        if self.evaluate_onset and self.evaluate_offset:
            title = "Event based metrics (onset-offset)"

        elif self.evaluate_onset and not self.evaluate_offset:
            title = "Event based metrics (onset only)"

        elif not self.evaluate_onset and self.evaluate_offset:
            title = "Event based metrics (offset only)"

        else:
            title = "Event based metrics"

        output = self.ui.section_header(title) + '\n'

        output += self.result_report_parameters() + '\n'
        output += self.result_report_overall() + '\n'
        output += self.result_report_class_wise_average() + '\n'
        output += self.result_report_class_wise() + '\n'

        return output
    def evaluate(self, reference_event_list, estimated_event_list):
        """Evaluate multiple files (reference and estimated)

        Parameters
        ----------
        reference_event_list : event list
            Reference event list

        estimated_event_list : event list
            Estimated event list

        Returns
        -------
        self

        """
        # Ensure input is MetaDataContainer
        if not isinstance(reference_event_list, dcase_util.containers.MetaDataContainer):
            reference_event_list = dcase_util.containers.MetaDataContainer(reference_event_list)

        if not isinstance(estimated_event_list, dcase_util.containers.MetaDataContainer):
            estimated_event_list = dcase_util.containers.MetaDataContainer(estimated_event_list)

        # Get unique files from both lists
        reference_files = reference_event_list.unique_files
        estimated_files = estimated_event_list.unique_files

        all_files = set(reference_files) | set(estimated_files)

        # Process each file individually
        for filename in all_files:
            # Get events for this file
            ref_events = reference_event_list.filter(filename=filename)
            est_events = estimated_event_list.filter(filename=filename)

            # If no events, create empty MetaDataContainer
            if ref_events is None:
                ref_events = dcase_util.containers.MetaDataContainer()
            if est_events is None:
                est_events = dcase_util.containers.MetaDataContainer()

            # Evaluate this file
            self.evaluate_file(ref_events, est_events)

        return self

    def evaluate_file(self, reference_event_list, estimated_event_list):
        """Evaluate single file (reference and estimated)

        Parameters
        ----------
        reference_event_list : event list
            Reference event list for a single file

        estimated_event_list : event list
            Estimated event list for a single file

        Returns
        -------
        None

        """
        # Evaluate only valid events
        valid_reference_event_list = dcase_util.containers.MetaDataContainer()
        for item in reference_event_list:
            if 'event_onset' in item and 'event_offset' in item and 'event_label' in item:
                valid_reference_event_list.append(item)
            elif 'onset' in item and 'offset' in item and 'event_label' in item:
                valid_reference_event_list.append(item)

        reference_event_list = valid_reference_event_list

        valid_estimated_event_list = dcase_util.containers.MetaDataContainer()
        for item in estimated_event_list:
            if 'event_onset' in item and 'event_offset' in item and 'event_label' in item:
                valid_estimated_event_list.append(item)
            elif 'onset' in item and 'offset' in item and 'event_label' in item:
                valid_estimated_event_list.append(item)

        estimated_event_list = valid_estimated_event_list

        # Update evaluated length and file count
        if len(reference_event_list) > 0:
            max_offset = max(event.offset for event in reference_event_list)
        else:
            # If reference event list is empty, set max_offset to 0
            max_offset = 0.0

        self.evaluated_length += max_offset
        self.evaluated_files += 1

        # Overall metrics
        Nsys = len(estimated_event_list)
        Nref = len(reference_event_list)

        if self.event_matching_type == 'optimal':
            # Optimal matching using bipartite graph
            label_hit_matrix = numpy.zeros((Nref, Nsys), dtype=bool)
            for j in range(Nref):
                for i in range(Nsys):
                    label_hit_matrix[j, i] = reference_event_list[j]['event_label'] == estimated_event_list[i]['event_label']

            hit_matrix = label_hit_matrix.copy()
            if self.evaluate_onset:
                onset_hit_matrix = numpy.zeros((Nref, Nsys), dtype=bool)
                for j in range(Nref):
                    for i in range(Nsys):
                        onset_hit_matrix[j, i] = self.validate_onset(
                            reference_event=reference_event_list[j],
                            estimated_event=estimated_event_list[i],
                            t_collar=self.t_collar
                        )
                hit_matrix &= onset_hit_matrix

            if self.evaluate_offset:
                offset_hit_matrix = numpy.zeros((Nref, Nsys), dtype=bool)
                for j in range(Nref):
                    for i in range(Nsys):
                        offset_hit_matrix[j, i] = self.validate_offset(
                            reference_event=reference_event_list[j],
                            estimated_event=estimated_event_list[i],
                            t_collar=self.t_collar,
                            percentage_of_length=self.percentage_of_length
                        )
                hit_matrix &= offset_hit_matrix

            # Perform bipartite matching
            hits = numpy.where(hit_matrix)
            G = {}
            for ref_i, est_i in zip(*hits):
                if est_i not in G:
                    G[est_i] = []
                G[est_i].append(ref_i)

            matching = sorted(utils.bipartite_match(G).items())
            Ntp = len(matching)

            # Mark matched reference and estimated events
            ref_matched = numpy.zeros(Nref, dtype=bool)
            est_matched = numpy.zeros(Nsys, dtype=bool)
            for ref_i, est_i in matching:
                ref_matched[ref_i] = True
                est_matched[est_i] = True

            # Calculate substitutions
            Nsubs = 0
            ref_unmatched_indices = numpy.where(~ref_matched)[0]
            est_unmatched_indices = numpy.where(~est_matched)[0]

            for ref_i in ref_unmatched_indices:
                ref_event = reference_event_list[ref_i]
                for est_i in est_unmatched_indices:
                    est_event = estimated_event_list[est_i]
                    onset_condition = self.validate_onset(
                        ref_event, est_event, self.t_collar) if self.evaluate_onset else True
                    offset_condition = self.validate_offset(
                        ref_event, est_event, self.t_collar, self.percentage_of_length) if self.evaluate_offset else True

                    if onset_condition and offset_condition:
                        Nsubs += 1
                        est_unmatched_indices = est_unmatched_indices[est_unmatched_indices != est_i]
                        break

        elif self.event_matching_type == 'greedy':
            # Greedy matching
            Ntp = 0
            Nsubs = 0
            est_matched = numpy.zeros(Nsys, dtype=bool)
            ref_matched = numpy.zeros(Nref, dtype=bool)

            for ref_i in range(Nref):
                ref_event = reference_event_list[ref_i]
                for est_i in range(Nsys):
                    if not est_matched[est_i]:
                        est_event = estimated_event_list[est_i]
                        label_condition = ref_event['event_label'] == est_event['event_label']
                        onset_condition = self.validate_onset(
                            ref_event, est_event, self.t_collar) if self.evaluate_onset else True
                        offset_condition = self.validate_offset(
                            ref_event, est_event, self.t_collar, self.percentage_of_length) if self.evaluate_offset else True

                        if label_condition and onset_condition and offset_condition:
                            Ntp += 1
                            est_matched[est_i] = True
                            ref_matched[ref_i] = True
                            break

            # Calculate substitutions
            ref_unmatched_indices = numpy.where(~ref_matched)[0]
            est_unmatched_indices = numpy.where(~est_matched)[0]

            for ref_i in ref_unmatched_indices:
                ref_event = reference_event_list[ref_i]
                for est_i in est_unmatched_indices:
                    est_event = estimated_event_list[est_i]
                    onset_condition = self.validate_onset(
                        ref_event, est_event, self.t_collar) if self.evaluate_onset else True
                    offset_condition = self.validate_offset(
                        ref_event, est_event, self.t_collar, self.percentage_of_length) if self.evaluate_offset else True

                    if onset_condition and offset_condition:
                        Nsubs += 1
                        est_unmatched_indices = est_unmatched_indices[est_unmatched_indices != est_i]
                        break

        Nfp = Nsys - Ntp - Nsubs
        Nfn = Nref - Ntp - Nsubs

        # Update overall metrics
        self.overall['Nref'] += Nref
        self.overall['Nsys'] += Nsys
        self.overall['Ntp'] += Ntp
        self.overall['Nsubs'] += Nsubs
        self.overall['Nfp'] += Nfp
        self.overall['Nfn'] += Nfn

        # Class-wise metrics
        for class_label in self.event_label_list:
            class_Nref = sum(1 for event in reference_event_list if event['event_label'] == class_label)
            class_Nsys = sum(1 for event in estimated_event_list if event['event_label'] == class_label)
            class_Ntp = 0

            if self.event_matching_type == 'optimal':
                class_ref_events = [event for event in reference_event_list if event['event_label'] == class_label]
                class_est_events = [event for event in estimated_event_list if event['event_label'] == class_label]
                Nref_c = len(class_ref_events)
                Nsys_c = len(class_est_events)

                hit_matrix = numpy.ones((Nref_c, Nsys_c), dtype=bool)
                if self.evaluate_onset:
                    onset_hit_matrix = numpy.zeros((Nref_c, Nsys_c), dtype=bool)
                    for j in range(Nref_c):
                        for i in range(Nsys_c):
                            onset_hit_matrix[j, i] = self.validate_onset(
                                class_ref_events[j], class_est_events[i], self.t_collar)
                    hit_matrix &= onset_hit_matrix

                if self.evaluate_offset:
                    offset_hit_matrix = numpy.zeros((Nref_c, Nsys_c), dtype=bool)
                    for j in range(Nref_c):
                        for i in range(Nsys_c):
                            offset_hit_matrix[j, i] = self.validate_offset(
                                class_ref_events[j], class_est_events[i], self.t_collar, self.percentage_of_length)
                    hit_matrix &= offset_hit_matrix

                hits = numpy.where(hit_matrix)
                G = {}
                for ref_i, est_i in zip(*hits):
                    if est_i not in G:
                        G[est_i] = []
                    G[est_i].append(ref_i)

                matching = sorted(utils.bipartite_match(G).items())
                class_Ntp = len(matching)

            elif self.event_matching_type == 'greedy':
                est_matched = numpy.zeros(class_Nsys, dtype=bool)
                for j, ref_event in enumerate(reference_event_list):
                    if ref_event['event_label'] == class_label:
                        for i, est_event in enumerate(estimated_event_list):
                            if est_event['event_label'] == class_label and not est_matched[i]:
                                onset_condition = self.validate_onset(
                                    ref_event, est_event, self.t_collar) if self.evaluate_onset else True
                                offset_condition = self.validate_offset(
                                    ref_event, est_event, self.t_collar, self.percentage_of_length) if self.evaluate_offset else True

                                if onset_condition and offset_condition:
                                    class_Ntp += 1
                                    est_matched[i] = True
                                    break

            class_Nfp = class_Nsys - class_Ntp
            class_Nfn = class_Nref - class_Ntp

            self.class_wise[class_label]['Nref'] += class_Nref
            self.class_wise[class_label]['Nsys'] += class_Nsys
            self.class_wise[class_label]['Ntp'] += class_Ntp
            self.class_wise[class_label]['Nfp'] += class_Nfp
            self.class_wise[class_label]['Nfn'] += class_Nfn

    @staticmethod
    def validate_onset(reference_event, estimated_event, t_collar=0.200):
        """Validate estimated event based on event onset

        Parameters
        ----------
        reference_event : dict
            Reference event.

        estimated_event: dict
            Estimated event.

        t_collar : float > 0, seconds
            Time collar within which the estimated onset has to be to be considered valid.
            Default value 0.2

        Returns
        -------
        bool

        """

        """
        if 'event_onset' in reference_event and 'event_onset' in estimated_event:
            return math.fabs(reference_event['event_onset'] - estimated_event['event_onset']) <= t_collar

        elif 'onset' in reference_event and 'onset' in estimated_event:
            return math.fabs(reference_event['onset'] - estimated_event['onset']) <= t_collar

        """

        # Detect field naming style used and validate onset
        ref_onset = reference_event.get('event_onset', reference_event.get('onset'))
        est_onset = estimated_event.get('event_onset', estimated_event.get('onset'))

        return abs(ref_onset - est_onset) <= t_collar

    @staticmethod
    def validate_offset(reference_event, estimated_event, t_collar=0.200, percentage_of_length=0.5):
        """Validate estimated event based on event offset

        Parameters
        ----------
        reference_event : dict
            Reference event.

        estimated_event : dict
            Estimated event.

        t_collar : float > 0, seconds
            Time collar within which the estimated offset has to be to be considered valid.
            Default value 0.2

        percentage_of_length : float in [0, 1]
            Percentage of the length within which the estimated offset has to be to be considered valid.
            Default value 0.5

        Returns
        -------
        bool

        """
        # Detect field naming style used and validate offset
        ref_onset = reference_event.get('event_onset', reference_event.get('onset'))
        ref_offset = reference_event.get('event_offset', reference_event.get('offset'))
        est_offset = estimated_event.get('event_offset', estimated_event.get('offset'))

        annotated_length = ref_offset - ref_onset
        allowed_offset = max(t_collar, percentage_of_length * annotated_length)

        return abs(ref_offset - est_offset) <= allowed_offset

    # ... [Other methods remain unchanged]

    # Include methods for resetting, computing metrics, and generating reports as in the original class.



    def overall_f_measure(self):
        """Overall f-measure metrics (f_measure, precision, and recall)

        Returns
        -------
        dict
            results in a dictionary format
        """

        if self.overall['Nsys'] == 0 and self.empty_system_output_handling == 'zero_score':
            precision = 0

        else:
            precision = metric.precision(
                Ntp=self.overall['Ntp'],
                Nsys=self.overall['Nsys']
            )

        recall = metric.recall(
            Ntp=self.overall['Ntp'],
            Nref=self.overall['Nref']
        )

        f_measure = metric.f_measure(
            precision=precision,
            recall=recall
        )

        return {
            'f_measure': f_measure,
            'precision': precision,
            'recall': recall
        }

    def overall_error_rate(self):
        """Overall error rate metrics (error_rate, substitution_rate, deletion_rate, and insertion_rate)

        Returns
        -------
        dict
            results in a dictionary format

        """

        substitution_rate = metric.substitution_rate(
            Nref=self.overall['Nref'],
            Nsubstitutions=self.overall['Nsubs']
        )

        deletion_rate = metric.deletion_rate(
            Nref=self.overall['Nref'],
            Ndeletions=self.overall['Nfn']
        )

        insertion_rate = metric.insertion_rate(
            Nref=self.overall['Nref'],
            Ninsertions=self.overall['Nfp']
        )

        error_rate = metric.error_rate(
            substitution_rate_value=substitution_rate,
            deletion_rate_value=deletion_rate,
            insertion_rate_value=insertion_rate
        )

        return {
            'error_rate': error_rate,
            'substitution_rate': substitution_rate,
            'deletion_rate': deletion_rate,
            'insertion_rate': insertion_rate
        }

    def class_wise_count(self, event_label):
        """Class-wise counts (Nref and Nsys)

        Returns
        -------
        dict
            results in a dictionary format

        """

        return {
            'Nref': self.class_wise[event_label]['Nref'],
            'Nsys': self.class_wise[event_label]['Nsys']
        }

    def class_wise_f_measure(self, event_label):
        """Class-wise f-measure metrics (f_measure, precision, and recall)

        Returns
        -------
        dict
            results in a dictionary format

        """
        if self.class_wise[event_label]['Nsys'] == 0 and self.empty_system_output_handling == 'zero_score':
            precision = 0

        else:
            precision = metric.precision(
                Ntp=self.class_wise[event_label]['Ntp'],
                Nsys=self.class_wise[event_label]['Nsys']
            )

        recall = metric.recall(
            Ntp=self.class_wise[event_label]['Ntp'],
            Nref=self.class_wise[event_label]['Nref']
        )

        f_measure = metric.f_measure(
            precision=precision,
            recall=recall
        )

        return {
            'f_measure': f_measure,
            'precision': precision,
            'recall': recall
        }

    def class_wise_error_rate(self, event_label):
        """Class-wise error rate metrics (error_rate, deletion_rate, and insertion_rate)

        Returns
        -------
        dict
            results in a dictionary format

        """

        deletion_rate = metric.deletion_rate(
            Nref=self.class_wise[event_label]['Nref'],
            Ndeletions=self.class_wise[event_label]['Nfn']
        )

        insertion_rate = metric.insertion_rate(
            Nref=self.class_wise[event_label]['Nref'],
            Ninsertions=self.class_wise[event_label]['Nfp']
        )

        error_rate = metric.error_rate(
            deletion_rate_value=deletion_rate,
            insertion_rate_value=insertion_rate
        )

        return {
            'error_rate': error_rate,
            'deletion_rate': deletion_rate,
            'insertion_rate': insertion_rate
        }

    # Reports
    def result_report_parameters(self):
        """Report metric parameters

        Returns
        -------
        str
            result report in string format

        """

        output = self.ui.data(field='Evaluated length', value=self.evaluated_length, unit='sec') + '\n'
        output += self.ui.data(field='Evaluated files', value=self.evaluated_files) + '\n'

        output += self.ui.data(field='Evaluate onset', value=self.evaluate_onset) + '\n'
        output += self.ui.data(field='Evaluate offset', value=self.evaluate_offset) + '\n'

        if self.t_collar < 1:
            output += self.ui.data(field='T collar', value=self.t_collar*1000, unit='ms') + '\n'

        else:
            output += self.ui.data(field='T collar', value=self.t_collar, unit='sec') + '\n'

        output += self.ui.data(field='Offset (length)', value=self.percentage_of_length*100, unit='%') + '\n'

        return output

    def reset(self):
        """Reset internal state
        """

        self.overall = {
            'Nref': 0.0,
            'Nsys': 0.0,
            'Nsubs': 0.0,
            'Ntp': 0.0,
            'Nfp': 0.0,
            'Nfn': 0.0,
        }
        self.class_wise = {}

        for class_label in self.event_label_list:
            self.class_wise[class_label] = {
                'Nref': 0.0,
                'Nsys': 0.0,
                'Ntp': 0.0,
                'Ntn': 0.0,
                'Nfp': 0.0,
                'Nfn': 0.0,
            }

        return self
