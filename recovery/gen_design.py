from itertools import combinations, permutations, product

import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 300)
pd.set_option('display.max_rows', 800)

class GenDesign:
    factor = 10
    n_blocks = 11
    ntrials_pre = (9, 12, 15, 18)
    obs_ratio = 1/3
    ntrials_init_block_obs = 3
    ntrials_noc = (0, 5, 10, 15)
    ntrials_prepost = 27
    n_stim = 5
    n_stim_presented = 2
    stimpool_size1 = 9999
    stimpool_size2 = 9999

    columns = dict(
        block='Int64',
        phase='Int64',
        trial='Int64',
        trial_running='Int64',
        trial_sync='Int64',
        trial_prepost='Int64',
        trial_type='Int64',
        trial_phase='Int64',
        outcome_shown='bool',
        pair='Int64',
        equal_value_pair='bool',
        omit_equal_phase='bool',
        noequal_value_pair='bool',
        type_choice='bool',
        type_choice_obs='bool',
        type_obs='bool',
        type_rating='bool',
        stimulus_left='Int64',
        stimulus_right='Int64',
        stimulus_left_outcome_id='Int64',
        stimulus_right_outcome_id='Int64',
        stimulus_id_left='Int64',
        stimulus_id_right='Int64',
        ntrials_pre='Int64',
        rating_pre='bool',
        rating_post='bool',
        ntrials_noc='Int64',
        type_choice_oc='bool',
        type_choice_noc='bool',
        type_choice_pre='bool',
        type_choice_post='bool',
        type_rating1='bool',
        type_rating2='bool',
        type_rating3='bool',
        stimulus_pool='Int64',
        outcome_schedule_id='Int64',
        outcome_base='Int64',
        outcome_diff='Int64',
        omit_equal_id='Int64',
        pre_equalshown_secondlasttrial='bool',
        pre_equalshown_lasttrial='bool',
        pre_equalnoshown_lasttrial_isobs='bool'
    )

    prominent_columns = ['block', 'phase', 'trial', 'trial_type', 'outcome_shown', 'pair', 'stimulus_left', 'stimulus_right']

    def __init__(self):

        self.other_columns = [key for key in self.columns.keys() if key not in self.prominent_columns]

        self.n_blocks *= self.factor

    def generate(self):

        # define length conditions of pre-noc choice phase - make sure that the four quarters of the experiment contain similar pre-noc block lenghts
        cond_ntrials_pre = np.array([np.random.permutation(len(self.ntrials_pre)) for _ in range(4*self.factor)]).flatten()[:self.n_blocks]

        # each block is now specificed with a tuple of length 3
        # 1st element (0-2): number of pre-noc trials is 5+2+2 (=0), 5+4+4 (=1), 5+5+5 (=2) or 5+6+6 (=3)
        # 2nd element (0-2): neither pre- nor post-noc rating (=0), only post-noc rating (=1), both pre- and post-noc rating (=2)
        # 3rd element (0-3): number of trials in the noc-phase (0=0, 6=1, 12=2, or 18=3)
        block_types = sum([list(product(range(3), range(4))) for _ in range(self.factor)], [])
        # remove the case of pre- and post-noc rating if there's no noc phase:
        for i in range(self.factor):
            block_types.remove((2, 0))
        # randomize order and add variable pre-noc trial numbers:
        block_types = [(cond_ntrials_pre[b], block_types[b][0], block_types[b][1]) for b in np.random.permutation(range(self.n_blocks))]

        block_ntrials_pre = [self.ntrials_pre[bt[0]] for bt in block_types]
        block_ntrials_pre_obs = [int((block_ntrials_pre[b] - self.ntrials_init_block_obs) * self.obs_ratio) for b in range(self.n_blocks)]
        block_ntrials_pre_choice = [block_ntrials_pre[b] - block_ntrials_pre_obs[b] for b in range(self.n_blocks)]
        block_ntrials_noc = [self.ntrials_noc[bt[2]] for bt in block_types]
        block_ntrials_post = [self.ntrials_prepost - block_ntrials_pre[b] for b in range(self.n_blocks)]
        block_ntrials_post_obs = [int(block_ntrials_post[b] * self.obs_ratio) for b in range(self.n_blocks)]
        block_ntrials_post_choice = [block_ntrials_post[b] - block_ntrials_post_obs[b] for b in range(self.n_blocks)]
        self.block_ntrials = [block_ntrials_pre[b] + (0, 0, self.n_stim)[bt[1]] + block_ntrials_noc[b] + (0, self.n_stim, self.n_stim)[bt[1]] + block_ntrials_post[b] + self.n_stim for b, bt in enumerate(block_types)]
        self.ntrials_total = np.sum(self.block_ntrials)

        outcome_schedule_order = np.array([np.random.permutation(range(self.n_stim - 1)) for _ in range(np.ceil(self.n_blocks/3).astype(int))]).flatten()[:self.n_blocks].tolist()
        possible_outcome_schedules = [tuple(np.sort(np.hstack((i, np.arange(1, self.n_stim) - int(self.n_stim / 2))))) for i in np.arange(1, self.n_stim) - int(self.n_stim / 2)]
        self.outcome_schedule = np.array([possible_outcome_schedules[i] for i in outcome_schedule_order])
        self.outcome_base = np.array([np.random.permutation((18, 23, 28)) for _ in range(np.ceil(self.n_blocks/3).astype(int))]).flatten()[:self.n_blocks]
        self.outcome_diff = np.array([np.random.permutation((3, 6)) for _ in range(np.ceil(self.n_blocks/2).astype(int))]).flatten()[:self.n_blocks]

        self.pairs = list(combinations(range(self.n_stim), self.n_stim_presented))
        n_pairs_total = len(self.pairs)
        pair_sides = list(permutations(range(self.n_stim_presented)))
        self.pairs_omitequal = [list(map(tuple, np.delete(self.pairs, self.pairs.index((r, r + 1)), axis=0).tolist())) for r in outcome_schedule_order]
        pair_ids_omitequal = [np.setdiff1d(range(n_pairs_total), self.pairs.index((r, r + 1))) for r in outcome_schedule_order]
        pair_ids_equal = [self.pairs.index((r, r + 1)) for r in outcome_schedule_order]
        pair_ids_noequal = [tuple([j for j, p in enumerate(self.pairs) if self.pairs[i][0] not in p and self.pairs[i][1] not in p]) for i in pair_ids_equal]
        self.stimuli_equal = [self.pairs[pair] for pair in pair_ids_equal]
        self.stimuli_noequal = [tuple(np.setdiff1d(range(self.n_stim), se)) for se in self.stimuli_equal]

        # 0: show equal-value pairs during pre, but not noc
        #    -> check whether noc phase can reverse equal-value pair choices (& what noc-aspects predict these reversals?)
        # 1: show equal-value pairs during noc, but not pre
        #    -> akin to CBIC-1 = check influence of $predictors on choices between equal-value options
        # 2: show equal-value pairs during pre and noc
        #    -> reversals during noc??
        # 3: show equal-value pairs during neither pre nor noc
        #    -> check whether certain factor such as the length of the nec-(or the pre-)phase influence the decision between equal-value options
        # note: equal-value pairs are always shown during post
        self.order_omitequal = np.array([np.random.permutation(range(4)) for _ in range(np.ceil(self.n_blocks/4).astype(int))]).flatten()[:self.n_blocks].tolist()
        self.pairs_pre = [(self.pairs, self.pairs_omitequal[b])[int(np.isin(oe, (1, 3)))] for b, oe in enumerate(self.order_omitequal)]
        pair_ids_pre = [(np.arange(n_pairs_total), pair_ids_omitequal[b])[int(np.isin(oe, (1, 3)))] for b, oe in enumerate(self.order_omitequal)]
        self.pairs_noc = [(self.pairs, self.pairs_omitequal[b])[int(np.isin(oe, (0, 3)))] for b, oe in enumerate(self.order_omitequal)]
        pair_ids_noc = [(np.arange(n_pairs_total), pair_ids_omitequal[b])[int(np.isin(oe, (0, 3)))] for b, oe in enumerate(self.order_omitequal)]
        n_pairs_pre = [len(p) for p in self.pairs_pre]
        n_pairs_noc = [len(p) for p in self.pairs_noc]

        self.stim_pool = np.array([np.random.permutation(range(2)) for _ in range(np.ceil(self.n_blocks/2).astype(int))]).flatten()[:self.n_blocks]
        self.stim_ids = np.zeros((self.n_blocks, self.n_stim), int)
        self.stim_ids[self.stim_pool == 0] = np.random.permutation(self.stimpool_size1)[:np.sum(self.stim_pool == 0) * self.n_stim].reshape(np.sum(self.stim_pool == 0), self.n_stim)
        self.stim_ids[self.stim_pool == 1] = np.random.permutation(self.stimpool_size2)[:np.sum(self.stim_pool == 1) * self.n_stim].reshape(np.sum(self.stim_pool == 1), self.n_stim)

        self.design = pd.DataFrame(columns=self.columns.keys(), index=range(self.ntrials_total)).astype(dtype=self.columns)
        for col in self.design.columns:
            if self.columns[col] == 'Int64':
                self.design[col] = pd.NA

        count_pre_init_fits, count_prepost_remain_fits, count_noc_fits = [], [], []

        for b in range(self.n_blocks):
            ntrials_until = np.sum(self.block_ntrials[:b], dtype=int)
            trange = np.arange(ntrials_until, ntrials_until + self.block_ntrials[b])
            self.design.loc[trange, 'trial_running'] = trange
            nsets_pre_choice = int(np.ceil(block_ntrials_pre_choice[b] / n_pairs_pre[b]))
            nsets_noc = int(np.ceil(block_ntrials_noc[b] / n_pairs_noc[b]))
            nsets_post_choice = int(np.ceil(block_ntrials_post_choice[b] / n_pairs_total))

            self.design.loc[trange, 'outcome_schedule_id'] = outcome_schedule_order[b]
            self.design.loc[trange, 'outcome_base'] = self.outcome_base[b]
            self.design.loc[trange, 'outcome_diff'] = self.outcome_diff[b]
            self.design.loc[trange, 'omit_equal_id'] = self.order_omitequal[b]

            self.design.loc[trange, 'block'] = b

            trial_types_pre_init = np.hstack((
                np.zeros(self.ntrials_init_block_obs, int),
                np.random.permutation(np.hstack((np.zeros(self.n_stim - self.ntrials_init_block_obs, int), 1)))
            ))
            trials_pre_remain = np.arange(self.n_stim + 1, block_ntrials_pre[b])
            ntrials_pre_remain = len(trials_pre_remain)
            ntrials_pre_remain_choice = int(ntrials_pre_remain * (1-self.obs_ratio))
            trials_noc = np.arange(block_ntrials_pre[b] + (0, self.n_stim)[block_types[b][1] == 2], block_ntrials_pre[b] + (0, self.n_stim)[block_types[b][1] == 2] + block_ntrials_noc[b])
            trials_post = np.arange(block_ntrials_pre[b] + (0, self.n_stim)[block_types[b][1] == 2] + block_ntrials_noc[b] + (0, self.n_stim)[block_types[b][1] > 0], self.block_ntrials[b]-self.n_stim)
            trials_prepost_remain = np.hstack((trials_pre_remain, trials_post))
            ntrials_prepost_remain = len(trials_prepost_remain)
            ntrials_post_choice = int((ntrials_prepost_remain - ntrials_pre_remain) * (1 - self.obs_ratio))


            ##  fill sequence-of-pairs array
            seq_pairs = np.full(self.block_ntrials[b], -1, dtype=int)

            # make sure each stimulus appears twice in the pre-init phase
            nofit, count = True, 0
            while nofit:
                count += 1
                stim1 = np.random.permutation(range(self.n_stim))
                pairids_pre_init = np.random.permutation([self.pairs.index(tuple(np.sort(p))) for p in np.stack((stim1, np.roll(stim1, 1))).T])
                # if the equal-value pair is presented in the pre-phase and if the whole set of pairs is at most
                # presented once, we spare the presentation of the equal value pair for the end of the pre-phase
                if bool(np.isin(self.order_omitequal[b], (0, 2))):
                    if not ((nsets_pre_choice == 1) and pair_ids_equal[b] in pairids_pre_init):
                        nofit = False
                else:
                    if not pair_ids_equal[b] in pairids_pre_init:
                        nofit = False
            count_pre_init_fits += [count]
            seq_pairs[np.where(trial_types_pre_init == 0)[0]] = pairids_pre_init

            # generate pair sequence for the remaining prepost-phase
            nofit, count = True, 0
            while nofit:
                count += 1
                trial_types_prepost_remain = np.array([np.random.permutation((0, 0, 1)) for _ in range(int(ntrials_prepost_remain * self.obs_ratio))]).flatten()

                seq_pairs_prepost_remain = np.hstack((np.random.choice(pair_ids_pre[b], ntrials_pre_remain_choice), pair_ids_equal[b], np.random.choice(range(n_pairs_total), ntrials_post_choice - 1)))
                seq_pairs_prepost_remain_full = np.full(ntrials_prepost_remain, -1)
                seq_pairs_prepost_remain_full[trial_types_prepost_remain == 0] = seq_pairs_prepost_remain
                seq_stimuli_prepost_remain = np.array([self.pairs[p] for p in seq_pairs_prepost_remain]).flatten()
                hist_stim = np.histogram(seq_stimuli_prepost_remain, range(self.n_stim + 1))[0]
                # first trial of post-phase must be a choice-trial
                if (trial_types_prepost_remain[ntrials_pre_remain] == 0) and \
                        (np.unique(seq_pairs_prepost_remain, return_counts=True)[1].max() <= 3) and \
                        np.all(hist_stim >= 4) and np.all(hist_stim <= 8):

                    if np.isin(self.order_omitequal[b], (0, 2)):  # show equal-value pair during pre
                        if (seq_pairs_prepost_remain_full[ntrials_pre_remain - 2] == pair_ids_equal[b]) and ((seq_pairs_prepost_remain_full[ntrials_pre_remain - 1] == -1) or np.isin(seq_pairs_prepost_remain_full[ntrials_pre_remain - 1], pair_ids_noequal[b])):
                            nofit = False
                    else:
                        #  if the equal-value pair is not shown during pre, make sure that ..
                        #  .. either the 2nd-last trial is an obs trial AND the last trial doesn't contain an equal-value stimulus
                        #  .. or, if the 2nd-last trial is no obs trial, that the last trial is then an obs trial
                        #  the point of all this is that we use obs trials towards the end of the pre-phase to make equal-value stimuli as close in value as possible
                        if ((seq_pairs_prepost_remain_full[ntrials_pre_remain - 2] == -1) and np.isin(seq_pairs_prepost_remain_full[ntrials_pre_remain - 1], pair_ids_noequal[b])) or ((seq_pairs_prepost_remain_full[ntrials_pre_remain - 2] != -1) and (seq_pairs_prepost_remain_full[ntrials_pre_remain - 1] == -1)):
                            nofit = False
            count_prepost_remain_fits += [count]
            seq_pairs[trials_prepost_remain[trial_types_prepost_remain == 0]] = seq_pairs_prepost_remain


            # fill noc trials
            nofit, count = True, 0
            if block_ntrials_noc[b]:
                while nofit:
                    count += 1
                    seq_pairs_noc = np.random.choice(pair_ids_noc[b], block_ntrials_noc[b])
                    seq_stimuli_noc = np.array([self.pairs[p] for p in seq_pairs_noc]).flatten()
                    hist_stim_noc = np.histogram(seq_stimuli_noc, range(self.n_stim + 1))[0]
                    if (np.unique(seq_pairs_noc, return_counts=True)[1].max() <= block_ntrials_noc[b]/self.n_stim) and \
                            np.all(hist_stim_noc >= int(block_ntrials_noc[b]/self.n_stim)) and np.all(hist_stim_noc <= 3*int(block_ntrials_noc[b]/self.n_stim)):
                        if np.isin(self.order_omitequal[b], (1, 2)):
                            #  if the equal-value pair is to be contained in the noc-phase, make sure it is
                            if np.isin(pair_ids_equal[b], seq_pairs_noc):
                                nofit = False
                        else:
                            nofit = False
                seq_pairs[trials_noc] = seq_pairs_noc
            count_noc_fits += [count]

            #  0 = pre + choice + outcome
            #  1 = pre + observation + outcome
            #  2 = rating1
            #  3 = choice + no outcome
            #  4 = rating2
            #  5 = post + choice + outcome
            #  6 = post + observation + outcome
            #  7 = rating3
            self.design.loc[trange, 'trial_type'] = np.hstack((
                trial_types_pre_init,
                trial_types_prepost_remain[:ntrials_pre_remain],
                2 * np.ones((0, self.n_stim)[block_types[b][1] == 2], int),
                3 * np.ones(block_ntrials_noc[b], int),
                4 * np.ones((0, self.n_stim)[block_types[b][1] > 0], int),
                trial_types_prepost_remain[ntrials_pre_remain:] + 5,
                7 * np.ones(self.n_stim, int)
            ))

            self.design.loc[trange, 'outcome_shown'] = self.design.loc[trange].trial_type.isin((0, 1, 5, 6))
            self.design.loc[trange, 'trial'] = range(self.block_ntrials[b])
            self.design.loc[trange, 'rating_pre'] = block_types[b][1] == 2
            self.design.loc[trange, 'rating_post'] = block_types[b][1] > 0
            self.design.loc[trange, 'ntrials_pre'] = block_ntrials_pre[b]
            self.design.loc[trange, 'ntrials_noc'] = block_ntrials_noc[b]
            self.design.loc[trange, 'pre_equalshown_secondlasttrial'] = np.isin(self.order_omitequal[b], (0, 2)) & (np.arange(self.block_ntrials[b]) == block_ntrials_pre[b] - 2)
            self.design.loc[trange, 'pre_equalshown_lasttrial'] = np.isin(self.order_omitequal[b], (0, 2)) & (np.arange(self.block_ntrials[b]) == block_ntrials_pre[b] - 1)
            self.design.loc[trange, 'pre_equalnoshown_lasttrial_isobs'] = np.isin(self.order_omitequal[b], (1, 3)) & (seq_pairs == -1) & ((np.arange(self.block_ntrials[b]) == block_ntrials_pre[b] - 1) | (np.arange(self.block_ntrials[b]) == block_ntrials_pre[b] - 2))

            self.design.loc[trange, 'pair'] = seq_pairs
            self.design.loc[trange, 'equal_value_pair'] = self.design.pair[trange] == pair_ids_equal[b]
            self.design.loc[trange, 'noequal_value_pair'] = self.design.pair[trange].isin(pair_ids_noequal[b])

            ##  fill stimulus array (2 columns representing stimulus left and right of fixation)
            seq_stim = np.zeros((self.block_ntrials[b], self.n_stim_presented), dtype=int)
            # choice trials
            for p in range(n_pairs_total):
                ind_pair = np.where(seq_pairs == p)[0]
                seq_stim[ind_pair] = self.pairs[p]
                #  randomize sides for each pair
                sides = np.array([np.random.permutation((0, 1)) for _ in range(int(np.ceil(len(ind_pair)/2)))]).flatten()[:len(ind_pair)]
                for i, ind in enumerate(ind_pair):
                    seq_stim[ind] = seq_stim[ind, pair_sides[sides[i]]]
            # rating trials:
            for trial_type in (2, 4, 7):
                if trial_type in self.design.trial_type[trange].unique():
                    seq_stim[np.array(self.design[(self.design.block == b) & (self.design.trial_type == trial_type)].trial, int)] = np.tile(np.random.permutation(range(self.n_stim)), (2, 1)).T

            # set observation trials to -1
            seq_stim[np.array(self.design[(self.design.block == b) & (self.design.trial_type == 1)].trial, int)] = -1
            seq_stim[np.array(self.design[(self.design.block == b) & (self.design.trial_type == 6)].trial, int)] = -1

            self.design.loc[trange, 'stimulus_left'] = seq_stim[:, 0]
            self.design.loc[trange, 'stimulus_right'] = seq_stim[:, 1]
            self.design.loc[trange[seq_stim[:, 0] != -1], 'stimulus_id_left'] = self.stim_ids[b][seq_stim[seq_stim[:, 0] != -1, 0]]
            self.design.loc[trange[seq_stim[:, 1] != -1], 'stimulus_id_right'] = self.stim_ids[b][seq_stim[seq_stim[:, 1] != -1, 1]]
            self.design.loc[trange[seq_stim[:, 0] != -1], 'stimulus_left_outcome_id'] = [self.outcome_schedule[b][stim] for stim in self.design.stimulus_left[trange[seq_stim[:, 0] != -1]]]
            self.design.loc[trange[seq_stim[:, 1] != -1], 'stimulus_right_outcome_id'] = [self.outcome_schedule[b][stim] for stim in self.design.stimulus_right[trange[seq_stim[:, 1] != -1]]]
            self.design.loc[trange, 'stimulus_pool'] = self.stim_pool[b]

            self.design.loc[trange, 'trial_sync'] = np.hstack((
                range(block_ntrials_pre[b]),
                range(max(block_ntrials_pre), max(block_ntrials_pre) + (0, self.n_stim)[block_types[b][1] == 2]),
                range(max(block_ntrials_pre) + self.n_stim, max(block_ntrials_pre) + self.n_stim + block_ntrials_noc[b]),
                range(max(block_ntrials_pre) + self.n_stim + max(block_ntrials_noc), max(block_ntrials_pre) + self.n_stim + max(block_ntrials_noc) + (0, self.n_stim)[block_types[b][1] > 0]),
                range(max(block_ntrials_pre) + self.n_stim + max(block_ntrials_noc) + self.n_stim, max(block_ntrials_pre) + self.n_stim + max(block_ntrials_noc) + self.n_stim + block_ntrials_post[b]),
                range(max(block_ntrials_pre) + self.n_stim + max(block_ntrials_noc) + self.n_stim + max(block_ntrials_post), max(block_ntrials_pre) + self.n_stim + max(block_ntrials_noc) + self.n_stim + max(block_ntrials_post) + self.n_stim)
            ))
            self.design.loc[(self.design.block == b) & self.design.trial_type.isin((0, 1, 5, 6)), 'trial_prepost'] = range(self.ntrials_prepost)

            for p, tt in enumerate([(0, 1), (3,), (5, 6)]):
                self.design.loc[(self.design.block == b) & self.design.trial_type.isin(tt), 'trial_phase'] = range(len(self.design[(self.design.block == b) & self.design.trial_type.isin(tt)]))

        #  0 = pre + choice + outcome
        #  1 = pre + observation + outcome
        #  2 = rating1
        #  3 = choice + no outcome
        #  4 = rating2
        #  5 = post + choice + outcome
        #  6 = post + observation + outcome
        #  7 = rating3
        self.design.type_choice = self.design.trial_type.isin((0, 3, 5))
        self.design.type_choice_obs = self.design.trial_type.isin((0, 1, 3, 5, 6))
        self.design.type_choice_pre = self.design.trial_type == 0
        self.design.type_choice_post = self.design.trial_type == 5
        self.design.type_obs = self.design.trial_type.isin((1, 6))
        self.design.type_choice_oc = self.design.trial_type.isin((0, 5))
        self.design.type_choice_noc = self.design.trial_type == 3
        self.design.type_rating = self.design.trial_type.isin((2, 4, 7))
        self.design.type_rating1 = self.design.trial_type == 2
        self.design.type_rating2 = self.design.trial_type == 4
        self.design.type_rating3 = self.design.trial_type == 7

        self.design.loc[self.design.trial_type.isin((0, 1)), 'phase'] = 0
        self.design.loc[self.design.trial_type == 3, 'phase'] = 1
        self.design.loc[self.design.trial_type.isin((5, 6)), 'phase'] = 2

        self.design.omit_equal_phase = False
        self.design.loc[(self.design.phase == 0), 'omit_equal_phase'] = self.design[(self.design.phase == 0)].omit_equal_id.isin((1, 3))
        self.design.loc[(self.design.phase == 1), 'omit_equal_phase'] = self.design[(self.design.phase == 1)].omit_equal_id.isin((0, 3))

        self.design.loc[self.design.type_rating, 'pair'] = np.nan
        return self.design


if __name__ == '__main__':
    # bandit = Bandit()
    # bandit = Bandit(special=True, n_trials=15)
    # bandit = BanditObserver(special=True, n_trials=15)
    # np.random.seed(4)
    design = GenDesign()
    d = design.generate()

    # for seed in range(100):
    #     print(seed)
    #     bandit = Bandit2(training_mode=False)
        # print([bandit.design[(bandit.design.omit_equal != 1) & (bandit.design.trial_running == bandit.design[(bandit.design.phase == 0) & (bandit.design.block == b)].trial_running.values[-1])].equal_value_pair.values[0] for b in np.where(np.isin(bandit.order_omitequal, (0, 2)))[0]])
        # print([bandit.design[bandit.design.trial_running == bandit.design[(bandit.design.phase == 2) & (bandit.design.block == b)].trial_running.values[0]].equal_value_pair.values[0] for b in range(11)])