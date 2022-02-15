setwd("C:/Users/esthe/Desktop/GLMM")

library(readr)
library(lmerTest)
library(lme4)
library(sjPlot)


################## read csv data files ############################

data_phases2_consistency <- read_csv("data_phase2_consistency.csv")
data_phases2_correct <- read_csv("data_phase2_correct.csv")
data_phases13_correct <- read_csv("data_phases13_correct.csv")


################## generalized linear mixed effects model (glmer) for performance measure (p1 - p3) ############################

model13.correct <- glmer(correct ~ block_difficulty+block_value_level+block_stimulus_type+block_ntrials_phase1+block_ntrials_phase2+trial_difficulty+trial_value_chosen+trial_number+(1|subject)+(1|block:subject),family='binomial', data=data_phases13_correct)
summary(model13.correct)
write.csv(summary(model13.correct)$coefficients, file = 'model_13_corr.csv')
tab_model(model13.correct)


################## glmer for performance measure (p2) ############################

model2.correct <- glmer(correct ~ block_difficulty+block_value_level+block_stimulus_type+block_ntrials_phase1+block_ntrials_phase2+trial_difficulty+trial_value_chosen+trial_number+(1|subject)+(1|block:subject),family='binomial', data=data_phases2_correct)
summary(model2.correct)
write.csv(summary(model2.correct)$coefficients, file = 'model_2_corr.csv')
tab_model(model2.correct)


################## glmer for choice consistency (p2) ############################

model2.consistency <- glmer(consistent ~ block_difficulty+block_value_level+block_stimulus_type+block_ntrials_phase1+block_ntrials_phase2+trial_difficulty+trial_value_chosen+trial_number+trial_pair_repeat_nr+(1|subject)+(1|block:subject),family='binomial', data=data_phases2_consistency)
summary(model2.consistency)
write.csv(summary(model2.consistency)$coefficients, file = 'model_2_consist.csv')
tab_model(model2.consistency)
