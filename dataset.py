import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    """
    Handles the Steinmetz dataset.
    It only considers sessions that have both
    visual and basal ganglia recordings.
    It only considers GO trials.
    When indexed, it only returns spikes (as tensors).
    """

    def __init__(
        self,
        data,
        visual_cortex_regions: list = [
            "VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],
        basal_ganglia_regions: list = [
            "ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"],
        threshold: int = 50,
    ) -> None:

        self.visual_cortex_regions = visual_cortex_regions
        self.basal_ganglia_regions = basal_ganglia_regions
        self.threshold = threshold

        self.data = self.visual_and_basal(data)
        self.data = self.get_go_trials(self.data)
        self.data = self.truncate(self.data)

        self.min_trials = self.find_min_dimensions(self.data)

    def __getitem__(
        self,
        session_index: int
    ) -> tuple:
        """
        Loads and returns a sample from the dataset at the given session_index.
        Based on the index, it identifies the visual and basal spks,
        handles them,
        converts them to tensors and returns the corresponding tuple.
        """

        session = self.data[session_index]

        spikes_visual = self.get_truncated_spikes(
            session, self.visual_cortex_regions)
        spikes_visual = torch.tensor(spikes_visual)
        spikes_visual = torch.permute(spikes_visual, (2, 1, 0))

        spikes_basal = self.get_truncated_spikes(
            session, self.basal_ganglia_regions)
        spikes_basal = torch.tensor(spikes_basal)
        spikes_basal = torch.permute(spikes_basal, (2, 1, 0))

        return spikes_visual, spikes_basal

    def __len__(self):
        return len(self.data)

    def visual_and_basal(
        self,
        data: np.array
    ) -> list:
        """
        It checks whether any of the considered brain areas recorded
        in the session is present in both visual or basal regions.
        """
        visual_and_basal_sessions = []
        for session in data:

            visual_ok = any(
                area in self.visual_cortex_regions
                for area in session["brain_area"])

            basal_ok = any(
                area in self.basal_ganglia_regions
                for area in session["brain_area"])

            if visual_ok and basal_ok:
                visual_and_basal_sessions.append(session)

        return visual_and_basal_sessions

    def truncate(
        self,
        data: list
    ) -> list:
        """
        Discards session with a number of neurons < self.threshold.
        """
        sessions_with_enough_neurons = []

        for session in data:
            vis_neuron_count = len(
                [i for i, area in enumerate(session["brain_area"])
                 if area in self.visual_cortex_regions])
            bg_neuron_count = len(
                [i for i, area in enumerate(session["brain_area"])
                 if area in self.basal_ganglia_regions])

            min_count = np.min([vis_neuron_count, bg_neuron_count])

            if min_count >= self.threshold:
                sessions_with_enough_neurons.append(session)

        return sessions_with_enough_neurons

    def get_go_trials(
        self,
        data: list
    ) -> list:
        """
        Within sessions, excludes no-go trials.
        """
        sessions_with_go_trials = []
        for session in data:
            go_trial_indices = [
                i for i, response in enumerate(session["response"])
                if response != 0]
            if go_trial_indices:
                filtered_session = {}  # sessions are dicts
                for key, value in session.items():
                    # Include all the keys of the session dictionary
                    if (isinstance(value, np.ndarray)
                            and value.ndim > 1
                            and value.shape[1] == len(session["response"])):
                        filtered_session[key] = value[:, go_trial_indices]
                    elif (isinstance(value, np.ndarray)
                            and len(value) == len(session["response"])):
                        filtered_session[key] = value[go_trial_indices]
                    else:  # like "mouse_name" is a single value
                        filtered_session[key] = value
                sessions_with_go_trials.append(filtered_session)

        return sessions_with_go_trials

    def find_min_dimensions(
        self,
        data: list
    ) -> tuple:
        """
        It finds the sesseion with the least number
        of trials across all session.
        """
        min_trials = float('inf')

        for session in data:
            spikes = session["spks"]  # [neurons_number, trial_Indexes, time]
            min_trials = min(min_trials, spikes.shape[1])

        return min_trials

    def get_truncated_spikes(
        self,
        session: dict,
        regions: list
    ) -> list:
        """
        It gets the "spks" from the dataset and
        reshape trials to min_trials and neurons to threshold.
        """
        spikes = session["spks"]
        spikes = spikes[:self.threshold, :, :]
        spikes = spikes[:, :self.min_trials, :]

        return spikes

    def __repr__(self):
        """
        Data summary, use print(dataset) to run.
        """

        num_sessions = len(self.data)
        num_sessions_string = f"Total number of sessions: {num_sessions}\n"
        truncating_string_trials = (
            f"We are truncating the number of trials across "
            f"all sessions to: {self.min_trials}\n")
        truncating_string_neurons = (
            f"We are truncating the number of neurons across "
            f"all sessions to: {self.threshold}\n")

        sep = " Sessions that meet criteria:\n"

        session_strings = []
        for n_session, session in enumerate(self.data):
            visual_neurons_shape = self.get_truncated_spikes(
                session, self.visual_cortex_regions).shape
            basal_neurons_shape = self.get_truncated_spikes(
                session, self.basal_ganglia_regions).shape
            session_string = (f"• Session {n_session}:\n"
                              f"\tvisual neurons: {visual_neurons_shape}\n"
                              f"\tbasal neurons: {basal_neurons_shape}")
            session_strings.append(session_string)

        all_session_strings = "\n".join(session_strings)

        string = (num_sessions_string +
                  truncating_string_trials +
                  truncating_string_neurons +
                  sep +
                  all_session_strings)

        return string


class Session(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        input_spikes = self.data[0][:, :, idx]
        label_action = self.data[1][idx]

        return (input_spikes.clone().detach().float(),
                label_action.clone().detach().long())


class DatasetMotion(Dataset):

    """
    I tried to make it tailored to our needs.
    It only considers sessions that have both
    visual and basal ganglia recordings.
    It only considers go trials.
    When indexed, it returns basal ganglia spikes and action).
    """

    def __init__(
        self,
        data,
        visual_cortex_regions: list = [
            "VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],
        basal_ganglia_regions: list = [
            "ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"],
        thresh: int = 50,
        augment_prob: float = 0.5,
        noise_level: float = 0.1
    ) -> None:

        self.visual_cortex_regions = visual_cortex_regions
        self.basal_ganglia_regions = basal_ganglia_regions
        self.thresh = thresh
        self.augment_prob = augment_prob
        self.noise_level = noise_level

        self.data = self.visual_and_basal(data)
        self.data = self.get_go_trials(self.data)
        self.min_trials, self.min_neurons = self.find_min_dimensions(self.data)

    def __getitem__(
        self,
        session_index: int
    ) -> torch.tensor:

        session = self.data[session_index]

        input_basal_spks = self.get_truncated_spikes(
            session, self.basal_ganglia_regions)
        label_action = session['response'][0:self.min_trials]

        input_basal_spks = input_basal_spks.clone().detach().float()
        label_action = torch.tensor(label_action, dtype=torch.long)

        return input_basal_spks, label_action

    def __len__(self):
        return len(self.data)

    def visual_and_basal(
        self,
        data: np.array
    ) -> list:
        """
        It checks whether any of the considered brain areas
        recorded in the session
        is present in both visual or basal regions.
        Also excludes sessions with a number of neurons < self.threshold
        """

        visual_and_basal_sessions = []
        for session in data:

            visual_ok = any(
                area in self.visual_cortex_regions
                for area in session["brain_area"])
            basal_ok = any(
                area in self.basal_ganglia_regions
                for area in session["brain_area"])
            vis_neuron_count = len(
                [i for i, area in enumerate(session["brain_area"])
                 if area in self.visual_cortex_regions])
            bg_neuron_count = len(
                [i for i, area in enumerate(session["brain_area"])
                 if area in self.basal_ganglia_regions])
            min_count = np.min([vis_neuron_count, bg_neuron_count])

            if visual_ok and basal_ok and min_count > self.thresh:
                visual_and_basal_sessions.append(session)

        return visual_and_basal_sessions

    def get_go_trials(
        self,
        data: list
    ) -> list:
        """
        Within sessions, excludes no-go trials.
        """
        sessions_with_go_trials = []
        for session in data:
            go_trial_indices = [
                i for i, response in enumerate(session["response"])
                if response != 0]
            if go_trial_indices:
                filtered_session = {}  # sessions are dicts
                for key, value in session.items():
                    # Include all the keys of the session dictionary
                    if (isinstance(value, np.ndarray)
                            and value.ndim > 1
                            and value.shape[1] == len(session["response"])):
                        filtered_session[key] = value[:, go_trial_indices]
                    elif (isinstance(value, np.ndarray)
                          and len(value) == len(session["response"])):
                        filtered_session[key] = value[go_trial_indices]
                    else:
                        filtered_session[key] = value
                sessions_with_go_trials.append(filtered_session)

        return sessions_with_go_trials

    def find_min_dimensions(
        self,
        data: list
    ) -> tuple:
        min_trials = float('inf')
        min_neurons = float('inf')
        """
        Helper function to establish """

        for session in data:
            spikes = session["spks"]  # [neurons_number, trial_Indexes, time]
            min_trials = min(min_trials, spikes.shape[1])
            # min_neurons = min(min_neurons, spikes.shape[0])

        return min_trials, min_neurons

    def get_truncated_spikes(
        self,
        session: dict,
        regions: list
    ) -> torch.tensor:

        neuron_indices = [
            i for i, area in enumerate(session["brain_area"])
            if area in regions]
        spikes = session["spks"][neuron_indices, :, :]
        spikes = spikes[:self.thresh, :, :]
        spikes = spikes[:, :self.min_trials, :]

        spikes = torch.tensor(spikes, dtype=torch.float32).clone().detach()
        spikes = torch.permute(spikes, (2, 0, 1))

        return spikes
