#!/usr/bin/env python3
import os
import copy
import fnmatch
import inspect
import pandas as pd
from matplotlib import pyplot as traceuse
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
import numpy as np
from scipy.stats import shapiro, anderson, normaltest, stats
import pingouin as statisticize
from statistics import mean, pstdev
import plotly.express as px
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

#starting task
START_TASK_DRAW = 1
START_TASK_HANOI = 2

# interruption task
START_INTERRUPTION_STROOP = 1
START_INTERRUPTION_MATH = 2

# hypotheses
HYPOTHESIS_SWITCH_TASK = 1
HYPOTHESIS_SWITCH_INTERRUPTION = 2


class DemographicData():
    def __init__(self):
        self.age = -1
        self.gender = ""
        self.education = ""

    def parse(self, pieces):
        self.age = pieces[3]
        self.gender = pieces[4]
        self.education = pieces[5]


class DiagnosisData():
    def __init__(self):
        self.asd = False
        self.color_blind = False
        self.hearing_impaired = False
        self.adhd = False
        self.prefer_not_to_say = False
        self.none = False

    def parse(self, pieces):
        self.asd = pieces[3]
        self.color_blind = pieces[4]
        self.hearing_impaired = pieces[5]
        self.adhd = pieces[6]
        self.prefer_not_to_say = pieces[7]
        self.none = pieces[8]


class EffortData():
    def __init__(self):
        self.task = ""
        self.effort = 0
        self.confidence = 0

    def parse(self, pieces):
        self.task = pieces[2]
        self.effort = pieces[3]
        self.confidence = pieces[4]


class SurveyData():
    def __init__(self):
        self.demographics = None
        self.diagnosis = None
        self.effort = []


class HanoiMove():
    def __init__(self):
        # ~ self.p1 = ""
        # ~ self.p2 = ""
        # ~ self.p3 = ""
        self.pegs = ""
        self.status = "incomplete"
        self.time = 0
        self.after_interruption = 0
        self.timeSpent = 0

    def parse(self, pieces, interruption_just_happened):
        self.pegs = pieces[3]
        self.status = pieces[5]
        self.time = pieces[8]
        self.after_interruption = interruption_just_happened
        self.timeSpent = pieces[6]



class HanoiTask():
    def __init__(self):
        self.hanoi_move_list = []
        self.time_to_complete = 0
        self.moves_to_complete = 0
        self.completed = False
        self.interrupted_during_Assessment = False
        self.interrupted_during_Training = False
        self.interrupted_during_Testing = False


class HanoiData():
    def __init__(self):
        self.hanoi_tasks = []
        self.average_time_move_piece = 0
        self.average_time_move_after_interruption = 0
        self.average_time_move_not_after_interruption = 0
        self.average_moves_to_complete = 0
        self.average_time_to_complete = 0


class DrawTask():
    def __init__(self):
        self.answer = ""
        self.correct_answer = ""
        self.percentage_correct = 0
        self.time = 0
        self.timeSpent = 0
        self.after_interruption = 0
        self.draw_response_list = []
        self.draw_tasks_responses = []
        self.interrupted_during_task = False

    def parse(self, pieces, interruption_just_happened):
        self.answer = pieces[3]
        self.correct_answer = pieces[4]
        self.percentage_correct = pieces[5]
        self.time = pieces[6]
        self.timeSpent = pieces[6]
        self.after_interruption = interruption_just_happened


class DrawData():
    def __init__(self):
        self.draw_tasks = []
        self.average_time_to_answer = 0
        self.average_time_to_answer_after_interruption = 0
        self.average_time_to_answer_after_no_interruption = 0
        self.average_correctness = 0
        self.average_correctness_after_interruption = 0
        self.average_correctness_after_no_interruption = 0
        self.averageTimeToAnswerDrawTaskEntirelyCorrect = 0


class StroopTask():
    def __init__(self):
        self.correct = False
        self.time = 0
        self.timeSpent = 0
        self.reTasked = 0
        self.stroopResponseList = []
        self.reTaskedDuringStroopTesting = False
        self.reTaskedDuringStroopTraining = False
        self.reTaskedDuringStroopAssessment = False

    def parse(self, pieces, tasked):
        if (pieces[5] == "CORRECT"):
            self.correct = True
        else:
            self.correct = False
        self.time = pieces[8]
        self.timeSpent = pieces[6]
        self.reTasked = tasked


class StroopData():
    def __init__(self):
        self.stroop_tasks = []
        self.totalTime = 0
        self.average_time = 0
        self.average_correctness = 0


class MathTask():
    def __init__(self):
        self.correct = False
        self.time = 0
        self.timeSpent = 0
        self.reTasked = 0
        self.mathResponseList = []
        self.reTasked_during_Testing = False
        self.reTasked_during_Training = False
        self.reTasked_during_Assessment = False

    def parse(self, pieces, tasked):
        if (pieces[5] == "CORRECT"):
            self.correct = True
        else:
            self.correct = False
        self.time = pieces[8]
        self.timeSpent = pieces[6]
        self.reTasked = tasked


class MathData():
    def __init__(self):
        self.math_tasks = []
        self.totalTime = 0
        self.average_time = 0
        self.average_correctness = 0


class Task():
    def __init__(self, name):
        self.name = name
        self.task = None


class Interruption():
    def __init__(self, name):
        self.name = name
        self.interruption = None


class Participant():
    def __init__(self, p_id):
        self.p_id = p_id
        self.group = 0

        self.starting_task = 0
        self.starting_interruption = 0
        self.hypotheses = 0
        self.survey = None

        self.tutorial_hanoi = None
        self.tutorial_draw = None
        self.tutorial_stroop = None
        self.tutorial_math = None

        self.assessment_task = None
        self.assessment_interruption = None
        self.training_task = None
        self.training_interruption = None
        self.testing_task = None
        self.testing_interruption = None

    def print_participant(self):
        pass


    def parse_condition(self, pieces):
        int_task = int(pieces[1])
        main_task = int(pieces[2])
        hypotheses = int(pieces[3])

        if (int_task == START_INTERRUPTION_MATH):
            self.starting_interruption = START_INTERRUPTION_MATH
        if (int_task == START_INTERRUPTION_STROOP):
            self.starting_interruption = START_INTERRUPTION_STROOP
        if (main_task == START_TASK_DRAW):
            self.starting_task = START_TASK_DRAW
        if (main_task == START_TASK_HANOI):
            self.starting_task = START_TASK_HANOI
        if (hypotheses == HYPOTHESIS_SWITCH_INTERRUPTION):
            self.hypotheses = HYPOTHESIS_SWITCH_INTERRUPTION
        if (hypotheses == HYPOTHESIS_SWITCH_TASK):
            self.hypotheses = HYPOTHESIS_SWITCH_TASK

def plotter(xAxis, lags, title, yLabel, PlotSpot, filenameForCharts,
            hanoiAccuracy, hanoiSpeed, drawAccuracy, drawSpeed,
            variationsEitherLagsStats, accuracyStats, speedStats):
    pass

    # # Demarcation of phases in plots
    # # x8 = [0, max(lags)]
    # x8 = [0, max(max(lags), max([x / 35 for x in drawAccuracy]), max([x / 60 for x in drawSpeed]), max([x / 60 for x in hanoiSpeed]), max([x / 60 for x in hanoiAccuracy]))]
    # y8 = [8.5, 8.5]
    # # x16 = [0, max(lags)]
    # x16 = [0, max(max(lags), max([x / 35 for x in drawAccuracy]), max([x / 60 for x in drawSpeed]), max([x / 60 for x in hanoiSpeed]), max([x / 60 for x in hanoiAccuracy]))]
    #
    # y16 = [16.5, 16.5]
    # x = [1, 12, 24]
    # traceuse.plot(y8, x8, color ='tab:red', linestyle="--")
    # traceuse.plot(y16, x16, color ='tab:red', linestyle="--")
    # # Obtain label for stats that are on chart
    # words = []
    # for characters in yLabel[8:24:1]:
    #     words.append(characters)
    #     # print(characters, end='')
    #     statsLabelOnChart = ''.join(words)
    # statsLabelOnChart+=" Significance:"
    #
    # # Plot part of Either Lags Stats on chart
    # if len(variationsEitherLagsStats.index) > 0:
    #     T, dof, pValue = round(variationsEitherLagsStats.iat[0, 0], 2), variationsEitherLagsStats.iat[0, 1], round(variationsEitherLagsStats.iat[0, 3], 3)
    #     if pValue < 0.001:
    #         pValue = "< 0.001"
    #         variationsEitherLagsStats = 't(' + str(dof) + ') = ' + str(T) + ', p ' + str(pValue)
    #     else:
    #         variationsEitherLagsStats = 't('+str(dof)+') = '+str(T)+', p = ' +str(pValue)
    #     traceuse.text(6.5, 1.75, statsLabelOnChart, ha='right', rotation=0, wrap=True, fontsize=6)
    #     traceuse.text(6.5, 1.5, variationsEitherLagsStats, ha='right', rotation=0, wrap=True, fontsize=6)
    #
    # # Plot Accuracy Significance on chart
    # if len(accuracyStats.index) > 0:
    #     T, dof, pValue = round(accuracyStats.iat[0, 0], 2), accuracyStats.iat[0, 1], round(accuracyStats.iat[0, 3], 3)
    #     if pValue < 0.001:
    #         pValue = "< 0.001"
    #         accuracyStats = 't(' + str(dof) + ') = ' + str(T) + ', p ' + str(pValue)
    #     else:
    #         accuracyStats = 't('+str(dof)+') = '+str(T)+', p = ' +str(pValue)
    #     traceuse.text(23.5, 1.75, 'Accuracy Significance:', ha='right', rotation=0, wrap=True, fontsize=6)
    #     traceuse.text(23.5, 1.5, accuracyStats, ha='right', rotation=0, wrap=True, fontsize=6)
    #
    # # Plot Speed Significance on Chart
    # if len(speedStats.index) > 0:
    #     T, dof, pValue = round(speedStats.iat[0, 0], 2), speedStats.iat[0, 1], round(speedStats.iat[0, 3], 3)
    #     if pValue < 0.001:
    #         pValue = "< 0.001"
    #         speedStats = 't(' + str(dof) + ') = ' + str(T) + ', p ' + str(pValue)
    #     else:
    #         speedStats = 't('+str(dof)+') = '+str(T)+', p = ' +str(pValue)
    #     traceuse.text(23.5, 1.0, 'Speed Significance:', ha='right', rotation=0, wrap=True, fontsize=6)
    #     traceuse.text(23.5, .75, speedStats, ha='right', rotation=0, wrap=True, fontsize=6)
    #
    # # Stopped here...need to reinstate training value in 'x' list for plotting in experiemtan condiiton
    # # Hanoi's Accuracy and Speed Charting
    # if p.group == 0 and p.starting_task == 2:
    #     del x[1]
    #     traceuse.plot(x, [element / 40 for element in hanoiAccuracy], color='blue', label = "Accuracy/Fewer Suboptimal Moves (Hanoi)", linestyle="-.")
    #     traceuse.plot(x, [element / 60 for element in hanoiSpeed], color='black', label = "Speed/Less Time to Complete (Hanoi)", linestyle=":")
    #     traceuse.legend(fontsize=5)
    #
    # elif p.group == 1 and p.starting_task == 2: # drop middle element, then plot
    #     del x[1]
    #     traceuse.plot(x, [element / 60 for element in hanoiAccuracy], color='blue', label = "Accuracy/Fewer Suboptimal Moves (Hanoi)", linestyle="-.")
    #     traceuse.plot(x, [element / 60 for element in hanoiSpeed], color='black', label = "Speed/Less Time to Complete (Hanoi)", linestyle=":")
    #     traceuse.legend(fontsize=5)
    #
    # # Draw's Accuracy and Speed Charting
    # elif p.group == 0 and p.starting_task == 1:
    #     del x[1]
    #     print("what's the values passed into elif 1?", drawAccuracy)
    #     traceuse.plot(x, [element * 3 for element in drawAccuracy], color='blue', label = "Accuracy/Order of Responses to Path-Tracing Task", linestyle="-.")
    #     traceuse.plot(x, [element / 30 for element in drawSpeed], color='black', label = "Speed/Less Time to Complete (Path-Tracing)", linestyle=":")
    #     traceuse.legend(fontsize=5)
    #
    # elif p.group == 1 and p.starting_task == 1: # drop middle element, then plot
    #     del x[1]
    #     traceuse.plot(x, [element * 5 for element in drawAccuracy], color='blue', label = "Accuracy/Order of Responses to Path-Tracing Task", linestyle="-.")
    #     traceuse.plot(x, [element / 35 for element in drawSpeed], color='black', label = "Speed/Less Time to Complete (Path-Tracing)", linestyle=":")
    #     traceuse.legend(fontsize=5)
    # else: pass
    #
    # traceuse.text(4.5, 0.15, "Assessment Phase", ha='center', rotation=0, wrap=True)
    # traceuse.text(12.5, 0.15, "Training Phase", ha='center', rotation=0, wrap=True)
    # traceuse.text(19.9, 0.15, "Testing Phase", ha='center', rotation=0, wrap=True)
    # xLags, yLags = range(1, len(xAxis) + 1), lags
    # traceuse.plot(xLags, yLags, color='green', label = yLabel)
    # traceuse.title(title)
    # traceuse.xlabel("24 Lag Times over Three Phases (Averages)", fontsize=11, wrap=True)
    # traceuse.xticks(fontsize=8, rotation=0)
    # traceuse.ylabel(yLabel)
    # traceuse.legend(fontsize=5, loc = 1)
    # # traceuse.yticks(fontsize=8, rotation=0)
    # traceuse.grid(color='gray', linestyle='--', linewidth=.15)
    # name = PlotSpot + filenameForCharts + ".pdf"
    # traceuse.savefig(name, bbox_inches='tight')
    # traceuse.show(block=False)
    # traceuse.pause(.1)
    # traceuse.close("all")
    return

def interConditionPlotter(xAxisExp,
                          ExperimentalLags,
                          xAxisControl,
                          ControlLags,
                          plotTitle,
                          yAxisLabel,
                          PlotPlace,
                          filenameForPlots,
                          significanceOfDifferenceStats):
    pass

    # Demarcation of phases in plots
    x8 = [0, max(max(ExperimentalLags), max(ControlLags))]
    y8 = [8.5, 8.5]
    x16 = [0, max(max(ExperimentalLags), max(ControlLags))]
    y16 = [16.5, 16.5]
    x = [1, 12, 24]
    traceuse.plot(y8, x8, color ='tab:red', linestyle="--")
    traceuse.plot(y16, x16, color ='tab:red', linestyle="--")
    # Obtain label for stats that are on chart
    words = []
    for characters in yAxisLabel[8:24:1]:
        words.append(characters)
        statsLabelOnPlot = ''.join(words)
        experimentalLabel = ''.join(words)
        controlLabel = ''.join(words)
    statsLabelOnPlot+=" Significance:"
    experimentalLabel += " (Experimental Intervention)"
    controlLabel += " (Control Comparison)"

    # Plot part of Either Lags Stats on chart
    if len(significanceOfDifferenceStats.index) > 0:
        T, dof, pValue = round(significanceOfDifferenceStats.iat[0, 1], 2), significanceOfDifferenceStats.iat[0, 2], round(significanceOfDifferenceStats.iat[0, 4], 2)
        if pValue < 0.001:
            pValue = "< 0.001"
            significanceOfDifferenceStats = 't(' + str(dof) + ') = ' + str(T) + ', p ' + str(pValue)
        else:
            significanceOfDifferenceStats = 't('+str(dof)+') = '+str(T)+', p = ' +str(pValue)
        traceuse.text(6.5, 1.75, statsLabelOnPlot, ha='right', rotation=0, wrap=True, fontsize=6)
        traceuse.text(6.5, 1.5, significanceOfDifferenceStats, ha='right', rotation=0, wrap=True, fontsize=6)

    traceuse.text(4.5, 0.15, "Assessment Phase", ha='center', rotation=0, wrap=True)
    traceuse.text(12.5, 0.15, "Training Phase", ha='center', rotation=0, wrap=True)
    traceuse.text(19.9, 0.15, "Testing Phase", ha='center', rotation=0, wrap=True)
    xAxisExp = range(1, len(xAxisExp) + 1)
    xAxisControl = range(1, len(xAxisControl) + 1)
    traceuse.plot(xAxisExp, ExperimentalLags, color='green', label = experimentalLabel)
    traceuse.plot(xAxisControl, ControlLags, color='blue', label = controlLabel)
    traceuse.title(plotTitle)
    traceuse.xlabel("24 Lag Times over Three Phases (Averages)", fontsize=11, wrap=True)
    traceuse.xticks(fontsize=8, rotation=0)
    traceuse.ylabel(yAxisLabel)
    traceuse.legend(fontsize=5, loc = 1)
    # traceuse.yticks(fontsize=8, rotation=0)
    traceuse.grid(color='gray', linestyle='--', linewidth=.15)
    name = PlotPlace + filenameForPlots + ".pdf"
    traceuse.savefig(name, bbox_inches='tight')
    traceuse.show(block=False)
    traceuse.pause(.1)
    traceuse.close("all")
    
    return


def intraConditionPlotter(xAxisExp,
                          ExperimentalLags,
                          xAxisControl,
                          ControlLags,
                          plotTitle,
                          yAxisLabel,
                          PlotPlace,
                          filenameForPlots,
                          significanceOfDifferenceStats,
                          xAxisText):
    pass

    # Obtain label for stats that are on chart
    words = []
    for characters in yAxisLabel[8:24:1]:
        words.append(characters)
        statsLabelOnPlot = ''.join(words)
        experimentalLabel = ''.join(words)
        controlLabel = ''.join(words)
    statsLabelOnPlot += " Significance:"
    experimentalLabel += " (Assessment Phase)"
    controlLabel += " (Testing Phase)"

    xAxisExp = range(1, len(xAxisExp) + 1)
    xAxisControl = range(1, len(xAxisControl) + 1)
    dfExp = pd.DataFrame(dict(a=xAxisExp, b=ExperimentalLags))
    dfControl = pd.DataFrame(dict(a=xAxisExp, b=ControlLags))
    slopeExp, interceptExp = np.polyfit(xAxisControl, ExperimentalLags, 1)
    regressesionLineExp = slopeExp * xAxisExp + interceptExp
    slopeControl, interceptControl = np.polyfit(xAxisControl, ControlLags, 1)
    regressesionLineControl = slopeControl * xAxisControl + interceptControl
    # lowessExp = lowess(ExperimentalLags, xAxisExp) # passed parameters in revers, returns both items
    lowessExpYaxis = lowess(ExperimentalLags, xAxisExp)[:, 1]  # obtain 'y' only
    lowessControlYaxis = lowess(ControlLags, xAxisControl)[:, 1]
    traceuse.plot(xAxisExp, lowessExpYaxis, color='green', label=experimentalLabel)
    traceuse.plot(xAxisControl, lowessControlYaxis, color='blue', label=controlLabel)

    # Plot part of Either Lags Stats on chart
    if len(significanceOfDifferenceStats.index) > 0:
        T, dof, pValue = round(significanceOfDifferenceStats.iat[0, 1], 2), significanceOfDifferenceStats.iat[
            0, 2], round(significanceOfDifferenceStats.iat[0, 4], 2)
        T, dof, pValue, timeSaved = round(significanceOfDifferenceStats.iat[0, 1], 2), \
                                    significanceOfDifferenceStats.iat[0, 2], \
                                    round(significanceOfDifferenceStats.iat[0, 4], 3), \
                                    round(significanceOfDifferenceStats.iat[0, 13], 2)
        if pValue < 0.001:
            pValue = "< 0.001"
            significanceOfDifferenceStats = 't(' + str(dof) + ') = ' + str(T) + ', p ' + str(pValue)
        else:
            significanceOfDifferenceStats = 't(' + str(dof) + ') = ' + str(T) + ', p = ' + str(pValue)

        if (max(max(lowessExpYaxis), max(lowessControlYaxis))) > 45:
            statsLabelOnPlotYaxis = (((max(max(lowessExpYaxis), max(lowessControlYaxis)))) - 9)
            significanceOfDifferenceStatsYaxis = (((max(max(lowessExpYaxis), max(lowessControlYaxis)))) - 10.5)
            timeSavedYaxis = (((max(max(lowessExpYaxis), max(lowessControlYaxis)))) - 12)
        elif (max(max(lowessExpYaxis), max(lowessControlYaxis))) > 14:
            statsLabelOnPlotYaxis = (((max(max(lowessExpYaxis), max(lowessControlYaxis)))) - 5)
            significanceOfDifferenceStatsYaxis = (((max(max(lowessExpYaxis), max(lowessControlYaxis)))) - 5.75)
            timeSavedYaxis = (((max(max(lowessExpYaxis), max(lowessControlYaxis)))) - 6.5)
        else:
            statsLabelOnPlotYaxis = ((max(max(lowessExpYaxis), max(lowessControlYaxis))) - .55)
            significanceOfDifferenceStatsYaxis = ((max(max(lowessExpYaxis), max(lowessControlYaxis))) - .65)
            timeSavedYaxis = ((max(max(lowessExpYaxis), max(lowessControlYaxis))) - .75)

        traceuse.text(5, statsLabelOnPlotYaxis, statsLabelOnPlot, fontsize=9)
        traceuse.text(5, significanceOfDifferenceStatsYaxis, significanceOfDifferenceStats,
                      fontsize=8, style='italic', fontweight='bold')
        traceuse.text(5, timeSavedYaxis,
                      "Time Savings: " + str(abs(timeSaved)) + " Seconds", fontsize=8, fontweight='bold')

    traceuse.title(plotTitle)
    traceuse.xlabel(str(len(xAxisExp)) + xAxisText, fontsize=11, wrap=True)
    traceuse.xticks(fontsize=8, rotation=0)
    traceuse.ylabel(yAxisLabel)
    traceuse.legend(fontsize=5, loc=1)
    # traceuse.yticks(fontsize=8, rotation=0)
    traceuse.grid(color='gray', linestyle='--', linewidth=.15)
    name = PlotPlace + filenameForPlots + ".pdf"
    traceuse.savefig(name, bbox_inches='tight')
    traceuse.show(block=False)
    traceuse.pause(.1)
    traceuse.close("all")

    return




def sortStackAverageStatisticize(assessInterruptLags,
                                 trainInterruptLags,
                                 testInterruptLags,
                                 assessResumptionLags,
                                 trainResumptionLags,
                                 testResumptionLags,
                                 collectSumResumptionLagsAssessment,
                                 collectSumResumptionLagsTesting,
                                 collectSumInterruptionLagsAssessment,
                                 collectSumInterruptionLagsTesting,
                                 stackedFlattenedAttentionList,
                                 stackedFlattenedResumptionList,
                                 accuracyInAssessmentSum,
                                 speedInAssessmentSum,
                                 accuracyInTestingSum,
                                 speedInTestingSum,
                                 accuraciesInAssessmentList,
                                 accuraciesInTestingList,
                                 speedsInAssessmentList,
                                 speedsInTestingList,
                                 collectSumsMovesAndSequencesAssessment,
                                 collectSumsMovesAndSequencesTesting,
                                 collectSumsCompletionTimesAssessment,
                                 collectSumsCompletionTimesTesting
                                 ):
    orderedDictListAttentions = \
        {
            'assessInterruptLags': assessInterruptLags,
            'trainingInterruptLags': trainInterruptLags,
            'testingInterruptLags': testInterruptLags
        }

    orderedDictListResumptions = \
        {
            'assessResumptionLagsList': assessResumptionLags,
            'trainingResumptionLagsList': trainResumptionLags,
            'testingResumptionLagsList': testResumptionLags
        }

    orderedDictListAccuracies = \
        {
            'accuracyInAssessmentList': accuraciesInAssessmentList,
            'accuracyInTestingList': accuraciesInTestingList
        }

    orderedDictListSpeeds = \
        {
            'speedInAssessmentList': speedsInAssessmentList,
            'speedInTestingList': speedsInTestingList
        }

    strippedDictAttention = list(orderedDictListAttentions.values())
    strippedDictResumption = list(orderedDictListResumptions.values())
    strippedDictListAccuracies = list(orderedDictListAccuracies.values())
    strippedDictListSpeeds = list(orderedDictListSpeeds.values())
    flattenedAttentionList = [item for sublist in strippedDictAttention for item in sublist]
    flattenedResumptionList = [item for sublist in strippedDictResumption for item in sublist]
    flattenedAccuraciesList = [item for sublist in strippedDictListAccuracies for item in sublist]
    flattenedSpeedsList = [item for sublist in strippedDictListSpeeds for item in sublist]

    stackedFlattenedAttentionList.append(flattenedAttentionList)
    # averages used for charting, not stats
    averageAttentions = [sum(allParticipantsAttentions) / len(stackedFlattenedAttentionList) for
                         allParticipantsAttentions in zip(*stackedFlattenedAttentionList)]

    stackedFlattenedResumptionList.append(flattenedResumptionList)
    # averages used for charting, not stats
    averageResumptions = [sum(allParticipantsResumptions) / len(stackedFlattenedResumptionList) for
                          allParticipantsResumptions in zip(*stackedFlattenedResumptionList)]

    stackedFlattenedAccuraciesList.append(flattenedAccuraciesList)
    # averages used for charting, not stats
    averageAccuracies = [sum(allParticipantsAccuracies) / len(stackedFlattenedAccuraciesList) for
                          allParticipantsAccuracies in zip(*stackedFlattenedAccuraciesList)]

    stackedFlattenedSpeedsList.append(flattenedSpeedsList)

    # averages used for charting, not stats
    averageSpeeds = [sum(allParticipantsSpeeds) / len(stackedFlattenedSpeedsList) for
                          allParticipantsSpeeds in zip(*stackedFlattenedSpeedsList)]


    # Statisticizing
    # Computing Resumption Lag Significances and differences btw phase's values
    # Sum per phase per participant used for stats
    collectSumResumptionLagsAssessment.append(sum(assessResumptionLags))
    collectSumResumptionLagsTesting.append(sum(testResumptionLags))

    differenceBtwAssessNTestResumption = [sumAssess - sumTest for (sumAssess, sumTest) in
                                          zip(collectSumResumptionLagsAssessment,
                                              collectSumResumptionLagsTesting)]

    # Computing Interruption Lag Significances and differences btw phase's values
    collectSumInterruptionLagsAssessment.append(sum(assessInterruptLags))
    collectSumInterruptionLagsTesting.append(sum(testInterruptLags))
    differenceBtwAssessNTestInterruption = [sumAssess - sumTest for (sumAssess, sumTest) in
                                          zip(collectSumInterruptionLagsAssessment,
                                              collectSumInterruptionLagsTesting)]

    # Computing Accuracy Significances and differences btw phase's values
    collectSumsMovesAndSequencesAssessment.append(accuracyInAssessmentSum)
    collectSumsMovesAndSequencesTesting.append(accuracyInTestingSum)
    differenceBtwAssessNTestAccuracy = [sumAssess - sumTest for (sumAssess, sumTest) in
                                            zip(collectSumsMovesAndSequencesAssessment,
                                                collectSumsMovesAndSequencesTesting)]

    # Computing Speed Significances and differences btw phase's values
    collectSumsCompletionTimesAssessment.append(speedInAssessmentSum)
    collectSumsCompletionTimesTesting.append(speedInTestingSum)
    differenceBtwAssessNTestSpeed = [sumAssess - sumTest for (sumAssess, sumTest) in
                                        zip(collectSumsCompletionTimesAssessment,
                                            collectSumsCompletionTimesTesting)]

    pingouinResumptionLagStatisticsResults = pd.DataFrame()
    pingouinInterruptionLagStatisticsResults = pd.DataFrame()
    pingouinAccuracyStatisticsResults = pd.DataFrame()
    pingouinSpeedStatisticsResults = pd.DataFrame()
    collectedSumResumptionLagsAssessment = []
    collectedSumResumptionLagsTesting = []
    collectedSumInterruptionLagsAssessment = []
    collectedSumInterruptionLagsTesting = []
    collectedSumsMovesAndSequencesAssessment = []
    collectedSumsMovesAndSequencesTesting = []
    collectedSumsCompletionTimesAssessment = []
    collectedSumsCompletionTimesTesting = []
    if len(collectSumResumptionLagsTesting) >= 15:
        # normality test for resumption lag variables
        stat, p = shapiro(collectSumResumptionLagsAssessment)
        # print('Normality Test Statistics = %.3f, p = %.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumResumptionLagsAssessment sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumResumptionLagsAssessment sample does NOT appear Gaussian (reject H0)')

        stat, p = shapiro(collectSumResumptionLagsTesting)
        # print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumResumptionLagsTesting sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumResumptionLagsTesting sample does NOT appear Gaussian (reject H0)')

        # normality test for interruption lag variables
        stat, p = shapiro(collectSumInterruptionLagsAssessment)
        # print('Normality Test Statistics = %.3f, p = %.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumInterruptionLagsAssessment sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumInterruptionLagsAssessment sample does NOT appear Gaussian (reject H0)')

        stat, p = shapiro(collectSumInterruptionLagsTesting)
        # print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumInterruptionLagsTesting sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumInterruptionLagsTesting sample does NOT appear Gaussian (reject H0)')

        # normality test for Accuracy variables
        stat, p = shapiro(collectSumsMovesAndSequencesAssessment)
        # print('Normality Test Statistics = %.3f, p = %.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsMovesAndSequencesAssessment sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsMovesAndSequencesAssessment sample does NOT appear Gaussian (reject H0)')

        stat, p = shapiro(collectSumsMovesAndSequencesTesting)
        # print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsMovesAndSequencesTesting sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsMovesAndSequencesTesting sample does NOT appear Gaussian (reject H0)')

        # normality test for Speed variables
        stat, p = shapiro(collectSumsCompletionTimesAssessment)
        # print('Normality Test Statistics = %.3f, p = %.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsCompletionTimesAssessment sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsCompletionTimesAssessment sample does NOT appear Gaussian (reject H0)')

        stat, p = shapiro(collectSumsCompletionTimesTesting)
        # print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsCompletionTimesTesting sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsCompletionTimesTesting sample does NOT appear Gaussian (reject H0)')

        # scipyStatisticsResults = stats.ttest_rel(collectSumResumptionLagsAssessment,
        #                                          collectSumResumptionLagsTesting)
        # print("statisticsResults: ", scipyStatisticsResults)
        # Resumption Lag Stats
        pingouinResumptionLagStatisticsResults = statisticize.ttest(collectSumResumptionLagsAssessment,
                                                 collectSumResumptionLagsTesting, paired=True, alternative="greater")

        # Interruption Lag Stats
        pingouinInterruptionLagStatisticsResults = statisticize.ttest(collectSumInterruptionLagsAssessment,
                                                  collectSumInterruptionLagsTesting, paired=True, alternative="greater")

        # Accuracy Stats
        pingouinAccuracyStatisticsResults = statisticize.ttest(collectSumsMovesAndSequencesAssessment,
                                              collectSumsMovesAndSequencesTesting, paired=True, alternative="greater")

        # Speed Stats
        pingouinSpeedStatisticsResults = statisticize.ttest(collectSumsCompletionTimesAssessment,
                                                collectSumsCompletionTimesTesting, paired=True, alternative="greater")
        collectedSumResumptionLagsAssessment = collectSumResumptionLagsAssessment
        collectedSumResumptionLagsTesting = collectSumResumptionLagsTesting

        collectedSumInterruptionLagsAssessment = collectSumInterruptionLagsAssessment
        collectedSumInterruptionLagsTesting = collectSumInterruptionLagsTesting

        collectedSumsMovesAndSequencesAssessment = collectSumsMovesAndSequencesAssessment
        collectedSumsMovesAndSequencesTesting = collectSumsMovesAndSequencesTesting

        collectedSumsCompletionTimesAssessment = collectSumsCompletionTimesAssessment
        collectedSumsCompletionTimesTesting = collectSumsCompletionTimesTesting

    return averageAttentions, \
           averageResumptions, \
           averageAccuracies, \
           averageSpeeds,\
           pingouinResumptionLagStatisticsResults, \
           pingouinInterruptionLagStatisticsResults,\
           pingouinAccuracyStatisticsResults,\
           pingouinSpeedStatisticsResults,\
           differenceBtwAssessNTestResumption,\
           differenceBtwAssessNTestInterruption,\
           differenceBtwAssessNTestAccuracy,\
           differenceBtwAssessNTestSpeed, \
           stackedFlattenedAttentionList,\
           stackedFlattenedResumptionList,\
           collectedSumResumptionLagsAssessment,\
           collectedSumResumptionLagsTesting,\
           collectedSumInterruptionLagsAssessment,\
           collectedSumInterruptionLagsTesting,\
           collectedSumsMovesAndSequencesAssessment,\
           collectedSumsMovesAndSequencesTesting,\
           collectedSumsCompletionTimesAssessment,\
           collectedSumsCompletionTimesTesting



def doAverages4SpeedAccuracyStatisticize(moveAssess,
                                moveTrain,
                                moveTest,
                                timeAssess,
                                timeTrain,
                                timeTest,
                                collectSumsMovesHanoiAssessment,
                                collectSumsMovesHanoiTraining,
                                collectSumsMovesHanoiTesting,
                                collectSumsCompletionTimeHanoiAssessment,
                                collectSumsCompletionTimeHanoiTraining,
                                collectSumsCompletionTimeHanoiTesting,
                                # moveDrawAssess,
                                # moveDrawTrain,
                                # moveDrawTest,
                                # timeDrawAssess,
                                # timeDrawTrain,
                                # timeDrawTest,
                                CollectCorrectnessDrawAssessment,
                                CollectCorrectnessDrawTraining,
                                CollectCorrectnessDrawTesting,
                                CollectSumsCompletionTimeDrawAssessment,
                                CollectSumsCompletionTimeDrawTraining,
                                CollectSumsCompletionTimeDrawTesting):
    collectSumsMovesHanoiAssessment.append(
        moveAssess)
    averagesOfSumsAssessment = sum(collectSumsMovesHanoiAssessment) / \
                              len(collectSumsMovesHanoiAssessment)
    collectSumsMovesHanoiTraining.append(
        moveTrain)
    averagesOfSumsTraining = sum(collectSumsMovesHanoiTraining) / \
                            len(collectSumsMovesHanoiTraining)
    collectSumsMovesHanoiTesting.append(
        moveTest)
    averagesOfSumsTesting = sum(collectSumsMovesHanoiTesting) / \
                           len(collectSumsMovesHanoiTesting)

    averagesMovesPerPhase_Accuracy = []
    averagesMovesPerPhase_Accuracy.append(averagesOfSumsAssessment)
    averagesMovesPerPhase_Accuracy.append(averagesOfSumsTraining)
    averagesMovesPerPhase_Accuracy.append(averagesOfSumsTesting)

    collectSumsCompletionTimeHanoiAssessment.append(
        timeAssess)
    averageOfSumsTimesAssessment = sum(collectSumsCompletionTimeHanoiAssessment) / \
                              len(collectSumsCompletionTimeHanoiAssessment)
    collectSumsCompletionTimeHanoiTraining.append(
        timeTrain)
    averageOfSumsTimesTraining = sum(collectSumsCompletionTimeHanoiTraining) / \
                            len(collectSumsCompletionTimeHanoiTraining)
    collectSumsCompletionTimeHanoiTesting.append(
        timeTest)
    averageOfSumsTimesTesting = sum(collectSumsCompletionTimeHanoiTesting) / \
                           len(collectSumsCompletionTimeHanoiTesting)

    averagesTimesPerPhase_Speed = []
    averagesTimesPerPhase_Speed.append(averageOfSumsTimesAssessment)
    averagesTimesPerPhase_Speed.append(averageOfSumsTimesTraining)
    averagesTimesPerPhase_Speed.append(averageOfSumsTimesTesting)

    # Statisticizing of Hanoi Accuracy and Speed
    pingouinHanoiAccuracyStatisticsResults = pd.DataFrame()
    pingouinHanoiSpeedStatisticsResults = pd.DataFrame()
    if len(collectSumsMovesHanoiAssessment) >= 15:
        # normality test for resumption lag variables
        stat, p = shapiro(collectSumsMovesHanoiAssessment)
        # print('Normality Test Statistics = %.3f, p = %.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsMovesHanoiAssessment sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsMovesHanoiAssessment sample does NOT appear Gaussian (reject H0)')

        stat, p = shapiro(collectSumsMovesHanoiTesting)
        # print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsMovesHanoiTesting sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsMovesHanoiTesting sample does NOT appear Gaussian (reject H0)')

        # normality test for interruption lag variables
        stat, p = shapiro(collectSumsCompletionTimeHanoiAssessment)
        # print('Normality Test Statistics = %.3f, p = %.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsCompletionTimeHanoiAssessment sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsCompletionTimeHanoiAssessment sample does NOT appear Gaussian (reject H0)')

        stat, p = shapiro(collectSumsCompletionTimeHanoiTesting)
        # print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsCompletionTimeHanoiTesting sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates collectSumsCompletionTimeHanoiTesting sample does NOT appear Gaussian (reject H0)')

        pingouinHanoiAccuracyStatisticsResults = statisticize.ttest(collectSumsMovesHanoiAssessment,
                                                              collectSumsMovesHanoiTesting, paired=True)

        pingouinHanoiSpeedStatisticsResults = statisticize.ttest(collectSumsCompletionTimeHanoiAssessment,
                                                                collectSumsCompletionTimeHanoiTesting, paired=True)

    # Demarcation...above Hanoi, below is for Draw

    CollectCorrectnessDrawAssessment.append(
        moveAssess)
    averagesOfSumsAssessmentDraw = sum(CollectCorrectnessDrawAssessment) / \
                              len(CollectCorrectnessDrawAssessment)
    CollectCorrectnessDrawTraining.append(
        moveTrain)
    averagesOfSumsTrainingDraw = sum(CollectCorrectnessDrawTraining) / \
                            len(CollectCorrectnessDrawTraining)
    CollectCorrectnessDrawTesting.append(
        moveTest)
    averagesOfSumsTestingDraw = sum(CollectCorrectnessDrawTesting) / \
                           len(CollectCorrectnessDrawTesting)
    averagesMovesPerPhaseDrawAccuracy = []
    averagesMovesPerPhaseDrawAccuracy.append(averagesOfSumsAssessmentDraw)
    averagesMovesPerPhaseDrawAccuracy.append(averagesOfSumsTrainingDraw)
    averagesMovesPerPhaseDrawAccuracy.append(averagesOfSumsTestingDraw)

    CollectSumsCompletionTimeDrawAssessment.append(
        timeAssess)
    averageOfSumsTimesAssessmentDraw = sum(CollectSumsCompletionTimeDrawAssessment) / \
                              len(CollectSumsCompletionTimeDrawAssessment)
    CollectSumsCompletionTimeDrawTraining.append(
        timeTrain)
    averageOfSumsTimesTrainingDraw = sum(CollectSumsCompletionTimeDrawTraining) / \
                            len(CollectSumsCompletionTimeDrawTraining)
    CollectSumsCompletionTimeDrawTesting.append(
        timeTest)
    averageOfSumsTimesTestingDraw = sum(CollectSumsCompletionTimeDrawTesting) / \
                           len(CollectSumsCompletionTimeDrawTesting)
    averagesTimesPerPhaseDrawSpeed = []
    averagesTimesPerPhaseDrawSpeed.append(averageOfSumsTimesAssessmentDraw)
    averagesTimesPerPhaseDrawSpeed.append(averageOfSumsTimesTrainingDraw)
    averagesTimesPerPhaseDrawSpeed.append(averageOfSumsTimesTestingDraw)

    # Statisticizing of Draw Speed and Accuracy
    pingouinDrawAccuracyStatisticsResults = pd.DataFrame()
    pingouinDrawSpeedStatisticsResults = pd.DataFrame()
    if len(CollectCorrectnessDrawAssessment) >= 15:
        # normality test for resumption lag variables
        stat, p = shapiro(CollectCorrectnessDrawAssessment)
        # print('Normality Test Statistics = %.3f, p = %.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates CollectCorrectnessDrawAssessment sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates CollectCorrectnessDrawAssessment sample does NOT appear Gaussian (reject H0)')

        stat, p = shapiro(CollectCorrectnessDrawTesting)
        # print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates CollectCorrectnessDrawTesting sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates CollectCorrectnessDrawTesting sample does NOT appear Gaussian (reject H0)')

        # normality test for interruption lag variables
        stat, p = shapiro(CollectSumsCompletionTimeDrawAssessment)
        # print('Normality Test Statistics = %.3f, p = %.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates CollectSumsCompletionTimeDrawAssessment sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates CollectSumsCompletionTimeDrawAssessment sample does NOT appear Gaussian (reject H0)')

        stat, p = shapiro(CollectSumsCompletionTimeDrawTesting)
        # print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            pass
            # print('Shapiro–Wilk test of normality indicates CollectSumsCompletionTimeDrawTesting sample appears Gaussian (fail to reject H0)')
        else:
            pass
            # print('Shapiro–Wilk test of normality indicates CollectSumsCompletionTimeDrawTesting sample does NOT appear Gaussian (reject H0)')

        pingouinDrawAccuracyStatisticsResults = statisticize.ttest(CollectCorrectnessDrawAssessment,
                                                              CollectCorrectnessDrawTesting, paired=True)

        pingouinDrawSpeedStatisticsResults = statisticize.ttest(CollectSumsCompletionTimeDrawAssessment,
                                                           CollectSumsCompletionTimeDrawTesting, paired=True)

    return averagesMovesPerPhase_Accuracy, \
           averagesTimesPerPhase_Speed, \
           averagesMovesPerPhaseDrawAccuracy, \
           averagesTimesPerPhaseDrawSpeed
           # pingouinHanoiAccuracyStatisticsResults,\
           # pingouinHanoiSpeedStatisticsResults,\
           # pingouinDrawAccuracyStatisticsResults,\
           # pingouinDrawSpeedStatisticsResults


def lineNumber():
    # Returns the current line number
    return inspect.currentframe().f_back.f_lineno


###############################################################################

directory = "../populationData/"
Matches = []
pid = []

averageTimeMathInterruptionsListAssess = []
averageTimeMathInterruptionsListTrain = []
averageTimeMathInterruptionsListTest = []

averageTimeStroopInterruptionsListAssess = []
averageTimeStroopInterruptionsListTrain = []
averageTimeStroopInterruptionsListTest = []

averageTimeToAnswerDrawTaskEntirelyCorrectListAssess = []
averageTimeToAnswerDrawTaskEntirelyCorrectListTrain = []
averageTimeToAnswerDrawTaskEntirelyCorrectListTest = []

averageNumberOfMovesBeforeCompleteForAllHanoiTasksListAssess = []
averageNumberOfMovesBeforeCompleteForAllHanoiTasksListTrain = []
averageNumberOfMovesBeforeCompleteForAllHanoiTasksListTest = []

avgTimesToCompletionForAllHanoiTasksListAssessment = []
avgTimesToCompletionForAllHanoiTasksListTraining = []
avgTimesToCompletionForAllHanoiTasksListTesting = []

averageTimeRespondAfterInterruptionListAssessment =[]
averageTimeRespondAfterInterruptionListTraining =[]
averageTimeRespondAfterInterruptionListTesting =[]

averageTimeMoveAfterInterruptionListAssessment =[]
averageTimeMoveAfterInterruptionListTraining =[]
averageTimeMoveAfterInterruptionListTesting =[]

all_participants = []
pattern = '*.txt'
files = list() #os.listdir(directory)
for (dirpath, dirnames, filenames) in os.walk(directory):
    files += [os.path.join(dirpath, file) for file in filenames]
Matches = fnmatch.filter(files, pattern)
for filenames in Matches:
    f = open(directory+filenames, "r")
    filename = os.path.basename(filenames)
    p_id = os.path.splitext(filename)[0]
    p = Participant(p_id)
    if "Control" in filenames:
        p.group = 1
    elif "Experimental" in filenames:
        p.group = 0
    pid.append(p.p_id)

    sv = SurveyData()

    ht = HanoiTask()

    h = HanoiData()
    d = DrawData()
    s = StroopData()
    m = MathData()

    SCENE_SURVEYS = 1
    SCENE_TUTORIAL = 2
    SCENE_ASSESSEMENT = 3
    SCENE_TRAINING = 4
    SCENE_TESTING = 5

    scene = SCENE_SURVEYS

    line_n = 0
    
    interruption_just_happened = 0
    tasked = 0
    for line in f:
        pieces = line.split(',')
        line_n += 1
        if (line_n == 2):  # the second line has the information about the condition
            p.parse_condition(pieces)

        if (pieces[0]) == "SURVEYS":
            if (pieces[1] == "SURVEY"):
                if (pieces[2] == "DEMOGRAPHICS"):
                    dd = DemographicData()
                    dd.parse(pieces)
                    sv.demographics = dd
                if (pieces[2] == "DIAGNOSIS"):
                    diag = DiagnosisData()
                    diag.parse(pieces)
                    sv.diagnosis = diag

        if (pieces[0] == "TUTORIAL"):
            if (scene != SCENE_TUTORIAL):
                scene = SCENE_TUTORIAL
            if (pieces[1] == "INTERRUPTION"):
                if (pieces[2] == "stroop"):
                    st = StroopTask()
                    st.parse(pieces, 0)
                    s.stroop_tasks.append(st)
                if (pieces[2] == "area"):
                    ma = MathTask()
                    ma.parse(pieces, 0)
                    m.math_tasks.append(ma)
            if (pieces[1] == "PRIMARY"):
                if (pieces[2] == "HANOI"):
                    han = HanoiMove()
                    han.parse(pieces, 0)
                    ht.hanoi_move_list.append(han)
                    if (han.status == "complete"):
                        # ~ print ("complete")
                        h.hanoi_tasks.append(ht)
                        ht = HanoiTask()
                if (pieces[2] == "path"):
                    dr = DrawTask()
                    dr.parse(pieces, 0)
                    dr.draw_response_list.append(dr)
                    if (dr.percentage_correct == "25%" or "50%" or "100%"):
                        dr.draw_tasks_responses.append(dr)


        if (pieces[0]) == "ASSESSMENT":
            if (scene != SCENE_ASSESSEMENT):
                scene = SCENE_ASSESSEMENT
                p.tutorial_hanoi = copy.deepcopy(h)
                p.tutorial_draw = copy.deepcopy(d)
                p.tutorial_stroop = copy.deepcopy(s)
                p.tutorial_math = copy.deepcopy(m)

                h = HanoiData()
                d = DrawData()
                s = StroopData()
                m = MathData()

            if (pieces[1] == "INTERRUPTION"):
                interruption_just_happened = 1
                if (pieces[2] == "stroop"):
                    st = StroopTask()
                    st.parse(pieces, tasked)
                    s.stroop_tasks.append(st)
                    st.stroopResponseList.append(st)
                    if (tasked == 1):
                        st.reTaskedDuringStroopTesting = True
                        st.reTaskedDuringStroopTraining = True
                        st.reTaskedDuringStroopAssessment = True
                if (pieces[2] == "area"):
                    ma = MathTask()
                    ma.parse(pieces, tasked)
                    m.math_tasks.append(ma)
                    ma.mathResponseList.append(ma)
                    if (tasked == 1):
                        ma.reTasked_during_Testing = True
                        ma.reTasked_during_Training = True
                        ma.reTasked_during_Assessment = True
                tasked = 0
            if (pieces[1] == "PRIMARY"):
                tasked = 1
                if (pieces[2] == "HANOI"):
                    han = HanoiMove()
                    han.parse(pieces, interruption_just_happened)
                    ht.hanoi_move_list.append(han)
                    if (interruption_just_happened == 1):
                        ht.interrupted_during_Assessment = True
                        ht.interrupted_during_Training = True
                        ht.interrupted_during_Testing = True
                    if (han.status == "complete"):
                        h.hanoi_tasks.append(ht)
                        ht = HanoiTask()
                if (pieces[2] == "path"):
                    dr = DrawTask()
                    dr.parse(pieces, interruption_just_happened)
                    d.draw_tasks.append(dr)
                    dr.draw_response_list.append(dr)
                    if (interruption_just_happened == 1):
                        dr.interrupted_during_task = True
                    if (dr.percentage_correct == "25%" or "50%" or "100%"):
                        dr.draw_tasks_responses.append(dr)

                interruption_just_happened = 0
            if (pieces[1] == "SURVEY"):
                ef = EffortData()
                ef.parse(pieces)
                sv.effort.append(ef)

        if (pieces[0] == "TRAINING"):
            if (scene != SCENE_TRAINING):
                scene = SCENE_TRAINING
                if (p.starting_task == START_TASK_DRAW):
                    t = Task("draw")
                    t.task = copy.deepcopy(d)
                    p.assessment_task = t
                if (p.starting_task == START_TASK_HANOI):
                    t = Task("hanoi")
                    t.task = copy.deepcopy(h)
                    p.assessment_task = t
                if (p.starting_interruption == START_INTERRUPTION_MATH):
                    i = Interruption("math")
                    i.interruption = copy.deepcopy(m)
                    p.assessment_interruption = i
                if (p.starting_interruption == START_INTERRUPTION_STROOP):
                    i = Interruption("stroop")
                    i.interruption = copy.deepcopy(s)
                    p.assessment_interruption = i

                h = HanoiData()
                d = DrawData()
                s = StroopData()
                m = MathData()

            if (pieces[1] == "INTERRUPTION"):
                interruption_just_happened = 1
                if (pieces[2] == "stroop"):
                    st = StroopTask()
                    st.parse(pieces, tasked)
                    s.stroop_tasks.append(st)
                    st.stroopResponseList.append(st)
                    if (tasked == 1):
                        st.reTaskedDuringStroopTesting = True
                        st.reTaskedDuringStroopTraining = True
                        st.reTaskedDuringStroopAssessment = True
                if (pieces[2] == "area"):
                    ma = MathTask()
                    ma.parse(pieces, tasked)
                    m.math_tasks.append(ma)
                    ma.mathResponseList.append(ma)
                    if (tasked == 1):
                        ma.reTasked_during_Testing = True
                        ma.reTasked_during_Training = True
                        ma.reTasked_during_Assessment = True
                tasked = 0
            if (pieces[1] == "PRIMARY"):
                tasked = 1
                if (pieces[2] == "HANOI"):
                    han = HanoiMove()
                    han.parse(pieces, interruption_just_happened)
                    ht.hanoi_move_list.append(han)
                    if (interruption_just_happened == 1):
                        ht.interrupted_during_Assessment = True
                        ht.interrupted_during_Training = True
                        ht.interrupted_during_Testing = True
                    if (han.status == "complete"):
                        h.hanoi_tasks.append(ht)
                        ht = HanoiTask()
                if (pieces[2] == "path"):
                    dr = DrawTask()
                    dr.parse(pieces, interruption_just_happened)
                    d.draw_tasks.append(dr)
                    dr.draw_response_list.append(dr)
                    if (interruption_just_happened == 1):
                        dr.interrupted_during_task = True
                    if (dr.percentage_correct == "25%" or "50%" or "100%"):
                        dr.draw_tasks_responses.append(dr)

                interruption_just_happened = 0
            if (pieces[1] == "SURVEY"):
                ef = EffortData()
                ef.parse(pieces)
                sv.effort.append(ef)

        if (pieces[0]) == "TESTING":
            if (scene != SCENE_TESTING):
                scene = SCENE_TESTING
                if (p.starting_task == START_TASK_DRAW and p.hypotheses == HYPOTHESIS_SWITCH_TASK):
                    t = Task("hanoi")
                    t.task = copy.deepcopy(h)
                    p.training_task = t
                if (p.starting_task == START_TASK_HANOI and p.hypotheses == HYPOTHESIS_SWITCH_TASK):
                    t = Task("draw")
                    t.task = copy.deepcopy(d)
                    p.training_task = t
                if (p.starting_task == START_TASK_DRAW and p.hypotheses == HYPOTHESIS_SWITCH_INTERRUPTION):
                    t = Task("draw")
                    t.task = copy.deepcopy(d)
                    p.training_task = t
                if (p.starting_task == START_TASK_HANOI and p.hypotheses == HYPOTHESIS_SWITCH_INTERRUPTION):
                    t = Task("hanoi")
                    t.task = copy.deepcopy(h)
                    p.training_task = t

                if (p.starting_interruption == START_INTERRUPTION_MATH and p.hypotheses == HYPOTHESIS_SWITCH_TASK):
                    i = Interruption("math")
                    i.interruption = copy.deepcopy(m)
                    p.training_interruption = i
                if (p.starting_interruption == START_INTERRUPTION_STROOP and p.hypotheses == HYPOTHESIS_SWITCH_TASK):
                    i = Interruption("stroop")
                    i.interruption = copy.deepcopy(s)
                    p.training_interruption = i
                if (p.starting_interruption == START_INTERRUPTION_STROOP and p.hypotheses == HYPOTHESIS_SWITCH_INTERRUPTION):
                    i = Interruption("math")
                    i.interruption = copy.deepcopy(m)
                    p.training_interruption = i
                if (p.starting_interruption == START_INTERRUPTION_MATH and p.hypotheses == HYPOTHESIS_SWITCH_INTERRUPTION):
                    i = Interruption("stroop")
                    i.interruption = copy.deepcopy(s)
                    p.training_interruption = i

                h = HanoiData()
                d = DrawData()
                s = StroopData()
                m = MathData()

            if (pieces[1] == "INTERRUPTION"):
                interruption_just_happened = 1
                if (pieces[2] == "stroop"):
                    st = StroopTask()
                    st.parse(pieces, tasked)
                    s.stroop_tasks.append(st)
                    st.stroopResponseList.append(st)
                    if (tasked == 1):
                        st.reTaskedDuringStroopTesting = True
                        st.reTaskedDuringStroopTraining = True
                        st.reTaskedDuringStroopAssessment = True
                if (pieces[2] == "area"):
                    ma = MathTask()
                    ma.parse(pieces, tasked)
                    m.math_tasks.append(ma)
                    ma.mathResponseList.append(ma)
                    if (tasked == 1):
                        ma.reTasked_during_Testing = True
                        ma.reTasked_during_Training = True
                        ma.reTasked_during_Assessment = True
                tasked = 0
            if (pieces[1] == "PRIMARY"):
                tasked = 1
                if (pieces[2] == "HANOI"):
                    han = HanoiMove()
                    han.parse(pieces, interruption_just_happened)
                    ht.hanoi_move_list.append(han)
                    if (interruption_just_happened == 1):
                        ht.interrupted_during_Assessment = True
                        ht.interrupted_during_Training = True
                        ht.interrupted_during_Testing = True
                    if (han.status == "complete"):
                        h.hanoi_tasks.append(ht)
                        ht = HanoiTask()
                if (pieces[2] == "path"):
                    dr = DrawTask()
                    dr.parse(pieces, interruption_just_happened)
                    d.draw_tasks.append(dr)
                    dr.draw_response_list.append(dr)
                    if (interruption_just_happened == 1):
                        dr.interrupted_during_task = True
                    if (dr.percentage_correct == "25%" or "50%" or "100%"):
                        dr.draw_tasks_responses.append(dr)
                interruption_just_happened = 0
            if (pieces[1] == "SURVEY"):
                ef = EffortData()
                ef.parse(pieces)
                sv.effort.append(ef)

    if (p.starting_task == START_TASK_DRAW):
        t = Task("draw")
        t.task = copy.deepcopy(d)
        p.testing_task = t
    if (p.starting_task == START_TASK_HANOI):
        t = Task("hanoi")
        t.task = copy.deepcopy(h)
        p.testing_task = t
    if (p.starting_interruption == START_INTERRUPTION_MATH):
        i = Interruption("math")
        i.interruption = copy.deepcopy(m)
        p.testing_interruption = i
    if (p.starting_interruption == START_INTERRUPTION_STROOP):
        i = Interruption("stroop")
        i.interruption = copy.deepcopy(s)
        p.testing_interruption = i
    p.survey = sv
    all_participants.append(p)

##############################################################################################

# PARTICIPANT DETAILS
id_arr = []
conditions_arr = []
control_arr = []
starting_task_arr = []
starting_interruption_arr = []
allParticipantsAttentionList = []
allParticipantsResumptionList = []

# DEMOGRAPHICS (d) VARIABLES
d_age = []
d_gender = []
d_education = []

d_asd = []
d_colorblind = []
d_hearingimpaired = []
d_adhd = []
d_prefernottosay = []
d_none = []

# EFFORT (e) VARIABLES
a_p_e_task = []
a_p_e_effort = []
a_p_e_confidence = []

a_i_e_task = []
a_i_e_effort = []
a_i_e_confidence = []

tr_p_e_task = []
tr_p_e_effort = []
tr_p_e_confidence = []

tr_i_e_task = []
tr_i_e_effort = []
tr_i_e_confidence = []

te_p_e_task = []
te_p_e_effort = []
te_p_e_confidence = []

te_i_e_task = []
te_i_e_effort = []
te_i_e_confidence = []

# ASSESSMENT (a) VARIABLES
a_i_name = []     # interruption (i) name
a_i_count = []    # total number of interruptions given
a_i_percentage = [] # percentage of average correctness across interruptions
a_i_time = []     # time during correct responses to interruptions
a_i_times = []    # aggregated time of average times for interruptions

a_p_name = []                 # primary (p) task name
a_p_count = []                # total number of tasks given
a_p_correctness = []          # weighted correctness across all tasks
a_p_time = []                 # time during correct responses to tasks
a_p_times = []                # average times to complete tasks
a_p_percentage = []           # percentage of average correctness across tasks (draw only)
a_p_percentage100 = []        # percentages of 100% correct responses to tasks (draw only)
a_p_resumption = []           # time to resume after interruption (hanoi only ??)
a_p_resumptions = []          # times to resume after interruptions (hanoi only ??)
a_p_interruptions = []        # total number of consective batch of interruptions during task (hanoi only)
a_p_movestotal = []           # total number of moves to complete all tasks (hanoi only)
a_p_movetasktime = []         # average time after a move (hanoi only)

# TRAINING (tr) VARIABLES
tr_i_name = []         # interruption name
tr_i_count = []        # total number of interruptions given
tr_i_percentage = []   # percentage of correct reposnses
tr_i_time = []         # time during correct responses to interruptions
tr_i_times = []        # aggregated time of average times for interruptions

tr_p_name = []                 # primary (p) task name
tr_p_count = []                # total number of tasks given
tr_p_correctness = []          # weighted correctness across all tasks
tr_p_time = []                 # time during correct responses to tasks
tr_p_times = []                # average times to complete tasks
tr_p_percentage = []           # percentage of average correctness across tasks (draw only)
tr_p_percentage100 = []        # percentages of 100% correct responses to tasks (draw only)
tr_p_resumption = []           # time to resume after interruption (hanoi only ??)
tr_p_resumptions = []          # times to resume after interruptions (hanoi only ??)
tr_p_interruptions = []        # total number of consective batch of interruptions during task (hanoi only)
tr_p_movestotal = []           # total number of moves to complete all tasks (hanoi only)
tr_p_movetasktime = []         # average time after a move (hanoi only)

# TESTING (te) VARIABLES
te_i_name = []         # interruption name
te_i_count = []        # total number of interruptions given
te_i_percentage = []   # percentage of correct reposnses
te_i_time = []         # time during correct responses to interruptions
te_i_times = []        # aggregated time of average times for interruptions

te_p_name = []                 # primary (p) task name
te_p_count = []                # total number of tasks given
te_p_correctness = []          # correctness across all tasks
te_p_time = []                 # time during correct responses to tasks
te_p_times = []                # average times to complete tasks
te_p_percentage = []           # percentage of average correctness across tasks (draw only)
te_p_percentage100 = []        # percentages of 100% correct responses to tasks (draw only)
te_p_resumption = []           # time to resume after interruption (hanoi only ??)
te_p_resumptions = []          # times to resume after interruptions (hanoi only ??)
te_p_interruptions = []        # total number of consective batch of interruptions during task (hanoi only)
te_p_movestotal = []           # total number of moves to complete all tasks (hanoi only)
te_p_movetasktime = []         # average time after a move (hanoi only)

# Functions enabling
flattenedAttentionList = []
flattenedResumptionList = []

collectSumsMovesHanoiAssessment = []
collectSumsMovesHanoiTraining = []
collectSumsMovesHanoiTesting = []

collectSumsCompletionTimeHanoiAssessment = []
collectSumsCompletionTimeHanoiTraining = []
collectSumsCompletionTimeHanoiTesting = []

CollectCorrectnessDrawAssessment = []
CollectCorrectnessDrawTraining = []
CollectCorrectnessDrawTesting = []

CollectSumsCompletionTimeDrawAssessment = []
CollectSumsCompletionTimeDrawTraining = []
CollectSumsCompletionTimeDrawTesting = []

stackedFlattenedAttentionList = []
stackedFlattenedResumptionList = []
stackedFlattenedAccuraciesList = []
stackedFlattenedSpeedsList = []

collectSumResumptionLagsAssessment = []
# collectSumResumptionLagsTraining = []
collectSumResumptionLagsTesting = []
differenceBtwAssessTestResumption = []

collectSumInterruptionLagsAssessment = []
# collectSumInterruptionLagsTesting = []
collectSumInterruptionLagsTesting = []
differenceBtwAssessTestInterruption = []

collectSumsMovesAndSequencesAssessment = []
collectSumsMovesAndSequencesTesting = []
differenceBtwAssessTestAccuracy = []

collectSumsCompletionTimesAssessment = []
collectSumsCompletionTimesTesting = []
differenceBtwAssessTestSpeed = []

# plot enabling list collection
stackedAttentionListExpH1 = []
stackedResumptionListExpH1 = []
stackedAttentionListExpH2 = []
stackedResumptionListExpH2 = []

stackedAttentionListControlH1 = []
stackedResumptionListControlH1 = []
stackedAttentionListControlH2 = []
stackedResumptionListControlH2 = []

numberOfMovesToCompleteHanoiStackedAssessment = []
numberOfMovesToCompleteHanoiStackedTesting = []

experimentalInterventionAttentions = []
experimentalInterventionResumptions = []
experimentalInterventionAccuracies = []
experimentalInterventionSpeeds = []

controlComparisonAttentions = []
controlComparisonResumptions = []
controlComparisonAccuracies = []
controlComparisonSpeeds = []

# Computation Enabling Lists
fewestHanoiMovesPerTaskAssessmentHardcoded = [1, 7, 7, 14, 13, 13, 7, 7, 3, 3, 14, 13, 3, 3, 7, 3]
fewestHanoiMovesPerTaskTestingHardcoded = [3, 7, 8, 3, 7, 7, 9, 2, 2, 7, 10, 7, 7, 3, 7, 7]


# Demarcation for Globally Declared/Localized Variational Use Containers
# ----------------------------------------------------------------------
# Globally Declared/Localized Variational Use: ExpH1DrawHanoiDrawStroop
ExpH1DrawHanoiDrawStroopSumResumptionLagsHanoiAssessment = []
ExpH1DrawHanoiDrawStroopSumResumptionLagsHanoiTesting = []

ExpH1DrawHanoiDrawStroopSumInterruptionLagsHanoiAssessment = []
ExpH1DrawHanoiDrawStroopSumInterruptionLagsHanoiTesting = []

ExpH1DrawHanoiDrawStroopstackedFlattenedAttentionList = []
ExpH1DrawHanoiDrawStroopstackedFlattenedResumptionList = []

# Hanoi Accuracy
ExpH1DrawHanoiDrawStroopcollectSumsMovesHanoiAssessment = []
ExpH1DrawHanoiDrawStroopcollectSumsMovesHanoiTraining = []
ExpH1DrawHanoiDrawStroopcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiAssessment = []
ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiTraining = []
ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ExpH1DrawHanoiDrawStroopCollectCorrectnessDrawAssessment = []
ExpH1DrawHanoiDrawStroopCollectCorrectnessDrawTraining = []
ExpH1DrawHanoiDrawStroopCollectCorrectnessDrawTesting = []
# Draw Speed
ExpH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawAssessment = []
ExpH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawTraining = []
ExpH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawTesting = []

ExpH1DrawHanoiDrawStroopcollectSumsMovesAndSequencesAssessment = []
ExpH1DrawHanoiDrawStroopcollectSumsMovesAndSequencesTesting = []

ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimesAssessment = []
ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimesTesting = []

ExpH1DrawHanoiDrawStroopDiffResumption = []
ExpH1DrawHanoiDrawStroopDiffInterruption = []
ExpH1DrawHanoiDrawStroopDiffAccuracy = []
ExpH1DrawHanoiDrawStroopDiffSpeed = []


# Globally Declared/Localized Variational Use: ExpH1DrawHanoiDrawMath
ExpH1DrawHanoiDrawMathSumResumptionLagsHanoiAssessment = []
ExpH1DrawHanoiDrawMathSumResumptionLagsHanoiTesting = []

ExpH1DrawHanoiDrawMathSumInterruptionLagsHanoiAssessment = []
ExpH1DrawHanoiDrawMathSumInterruptionLagsHanoiTesting = []

ExpH1DrawHanoiDrawMathstackedFlattenedAttentionList = []
ExpH1DrawHanoiDrawMathstackedFlattenedResumptionList = []

# Hanoi Accuracy
ExpH1DrawHanoiDrawMathcollectSumsMovesHanoiAssessment = []
ExpH1DrawHanoiDrawMathcollectSumsMovesHanoiTraining = []
ExpH1DrawHanoiDrawMathcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ExpH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiAssessment = []
ExpH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiTraining = []
ExpH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ExpH1DrawHanoiDrawMathCollectCorrectnessDrawAssessment = []
ExpH1DrawHanoiDrawMathCollectCorrectnessDrawTraining = []
ExpH1DrawHanoiDrawMathCollectCorrectnessDrawTesting = []
# Draw Speed
ExpH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawAssessment = []
ExpH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawTraining = []
ExpH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawTesting = []

ExpH1DrawHanoiDrawMathcollectSumsMovesAndSequencesAssessment = []
ExpH1DrawHanoiDrawMathcollectSumsMovesAndSequencesTesting = []

ExpH1DrawHanoiDrawMathcollectSumsCompletionTimesAssessment = []
ExpH1DrawHanoiDrawMathcollectSumsCompletionTimesTesting = []

ExpH1DrawHanoiDrawMathDiffResumption = []
ExpH1DrawHanoiDrawMathDiffInterruption = []
ExpH1DrawHanoiDrawMathDiffAccuracy = []
ExpH1DrawHanoiDrawMathDiffSpeed = []

# Globally Declared/Localized Variational Use: ExpH1HanoiDrawHanoiStroop
ExpH1HanoiDrawHanoiStroopSumResumptionLagsHanoiAssessment = []
ExpH1HanoiDrawHanoiStroopSumResumptionLagsHanoiTesting = []

ExpH1HanoiDrawHanoiStroopSumInterruptionLagsHanoiAssessment = []
ExpH1HanoiDrawHanoiStroopSumInterruptionLagsHanoiTesting = []

ExpH1HanoiDrawHanoiStroopstackedFlattenedAttentionList = []
ExpH1HanoiDrawHanoiStroopstackedFlattenedResumptionList = []

# Hanoi Accuracy
ExpH1HanoiDrawHanoiStroopcollectSumsMovesHanoiAssessment = []
ExpH1HanoiDrawHanoiStroopcollectSumsMovesHanoiTraining = []
ExpH1HanoiDrawHanoiStroopcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiAssessment = []
ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiTraining = []
ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ExpH1HanoiDrawHanoiStroopCollectCorrectnessDrawAssessment = []
ExpH1HanoiDrawHanoiStroopCollectCorrectnessDrawTraining = []
ExpH1HanoiDrawHanoiStroopCollectCorrectnessDrawTesting = []
# Draw Speed
ExpH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawAssessment = []
ExpH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawTraining = []
ExpH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawTesting = []

ExpH1HanoiDrawHanoiStroopcollectSumsMovesAndSequencesAssessment = []
ExpH1HanoiDrawHanoiStroopcollectSumsMovesAndSequencesTesting = []

ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimesAssessment = []
ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimesTesting = []

ExpH1HanoiDrawHanoiStroopDiffResumption = []
ExpH1HanoiDrawHanoiStroopDiffInterruption = []
ExpH1HanoiDrawHanoiStroopDiffAccuracy = []
ExpH1HanoiDrawHanoiStroopDiffSpeed = []


# Globally Declared/Localized Variational Use: ExpH1HanoiDrawHanoiMath
ExpH1HanoiDrawHanoiMathSumResumptionLagsHanoiAssessment = []
ExpH1HanoiDrawHanoiMathSumResumptionLagsHanoiTesting = []

ExpH1HanoiDrawHanoiMathSumInterruptionLagsHanoiAssessment = []
ExpH1HanoiDrawHanoiMathSumInterruptionLagsHanoiTesting = []

ExpH1HanoiDrawHanoiMathstackedFlattenedAttentionList = []
ExpH1HanoiDrawHanoiMathstackedFlattenedResumptionList = []

# Hanoi Accuracy
ExpH1HanoiDrawHanoiMathcollectSumsMovesHanoiAssessment = []
ExpH1HanoiDrawHanoiMathcollectSumsMovesHanoiTraining = []
ExpH1HanoiDrawHanoiMathcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiAssessment = []
ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiTraining = []
ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ExpH1HanoiDrawHanoiMathCollectCorrectnessDrawAssessment = []
ExpH1HanoiDrawHanoiMathCollectCorrectnessDrawTraining = []
ExpH1HanoiDrawHanoiMathCollectCorrectnessDrawTesting = []
# Draw Speed
ExpH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawAssessment = []
ExpH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawTraining = []
ExpH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawTesting = []

ExpH1HanoiDrawHanoiMathcollectSumsMovesAndSequencesAssessment = []
ExpH1HanoiDrawHanoiMathcollectSumsMovesAndSequencesTesting = []

ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimesAssessment = []
ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimesTesting = []

ExpH1HanoiDrawHanoiMathDiffResumption = []
ExpH1HanoiDrawHanoiMathDiffInterruption = []
ExpH1HanoiDrawHanoiMathDiffAccuracy = []
ExpH1HanoiDrawHanoiMathDiffSpeed = []

# Globally Declared/Localized Variational Use: ExpH2StroopMathStroopDraw
ExpH2StroopMathStroopDrawSumResumptionLagsHanoiAssessment = []
ExpH2StroopMathStroopDrawSumResumptionLagsHanoiTesting = []

ExpH2StroopMathStroopDrawSumInterruptionLagsHanoiAssessment = []
ExpH2StroopMathStroopDrawSumInterruptionLagsHanoiTesting = []

ExpH2StroopMathStroopDrawstackedFlattenedAttentionList = []
ExpH2StroopMathStroopDrawstackedFlattenedResumptionList = []

# Hanoi Accuracy
ExpH2StroopMathStroopDrawcollectSumsMovesHanoiAssessment = []
ExpH2StroopMathStroopDrawcollectSumsMovesHanoiTraining = []
ExpH2StroopMathStroopDrawcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ExpH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiAssessment = []
ExpH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiTraining = []
ExpH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ExpH2StroopMathStroopDrawCollectCorrectnessDrawAssessment = []
ExpH2StroopMathStroopDrawCollectCorrectnessDrawTraining = []
ExpH2StroopMathStroopDrawCollectCorrectnessDrawTesting = []
# Draw Speed
ExpH2StroopMathStroopDrawCollectSumsCompletionTimeDrawAssessment = []
ExpH2StroopMathStroopDrawCollectSumsCompletionTimeDrawTraining = []
ExpH2StroopMathStroopDrawCollectSumsCompletionTimeDrawTesting = []

ExpH2StroopMathStroopDrawcollectSumsMovesAndSequencesAssessment = []
ExpH2StroopMathStroopDrawcollectSumsMovesAndSequencesTesting = []

ExpH2StroopMathStroopDrawcollectSumsCompletionTimesAssessment = []
ExpH2StroopMathStroopDrawcollectSumsCompletionTimesTesting = []

ExpH2StroopMathStroopDrawDiffResumption = []
ExpH2StroopMathStroopDrawDiffInterruption = []
ExpH2StroopMathStroopDrawDiffAccuracy = []
ExpH2StroopMathStroopDrawDiffSpeed = []


# Globally Declared/Localized Variational Use: ExpH2MathStroopMathDraw
ExpH2MathStroopMathDrawSumResumptionLagsHanoiAssessment = []
ExpH2MathStroopMathDrawSumResumptionLagsHanoiTesting = []

ExpH2MathStroopMathDrawSumInterruptionLagsHanoiAssessment = []
ExpH2MathStroopMathDrawSumInterruptionLagsHanoiTesting = []

ExpH2MathStroopMathDrawstackedFlattenedAttentionList = []
ExpH2MathStroopMathDrawstackedFlattenedResumptionList = []

# Hanoi Accuracy
ExpH2MathStroopMathDrawcollectSumsMovesHanoiAssessment = []
ExpH2MathStroopMathDrawcollectSumsMovesHanoiTraining = []
ExpH2MathStroopMathDrawcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ExpH2MathStroopMathDrawcollectSumsCompletionTimeHanoiAssessment = []
ExpH2MathStroopMathDrawcollectSumsCompletionTimeHanoiTraining = []
ExpH2MathStroopMathDrawcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ExpH2MathStroopMathDrawCollectCorrectnessDrawAssessment = []
ExpH2MathStroopMathDrawCollectCorrectnessDrawTraining = []
ExpH2MathStroopMathDrawCollectCorrectnessDrawTesting = []
# Draw Speed
ExpH2MathStroopMathDrawCollectSumsCompletionTimeDrawAssessment = []
ExpH2MathStroopMathDrawCollectSumsCompletionTimeDrawTraining = []
ExpH2MathStroopMathDrawCollectSumsCompletionTimeDrawTesting = []

ExpH2MathStroopMathDrawcollectSumsMovesAndSequencesAssessment = []
ExpH2MathStroopMathDrawcollectSumsMovesAndSequencesTesting = []

ExpH2MathStroopMathDrawcollectSumsCompletionTimesAssessment = []
ExpH2MathStroopMathDrawcollectSumsCompletionTimesTesting = []

ExpH2MathStroopMathDrawDiffResumption = []
ExpH2MathStroopMathDrawDiffInterruption = []
ExpH2MathStroopMathDrawDiffAccuracy = []
ExpH2MathStroopMathDrawDiffSpeed = []

# Globally Declared/Localized Variational Use: ExpH2StroopMathStroopHanoi
ExpH2StroopMathStroopHanoiSumResumptionLagsHanoiAssessment = []
ExpH2StroopMathStroopHanoiSumResumptionLagsHanoiTesting = []

ExpH2StroopMathStroopHanoiSumInterruptionLagsHanoiAssessment = []
ExpH2StroopMathStroopHanoiSumInterruptionLagsHanoiTesting = []

ExpH2StroopMathStroopHanoistackedFlattenedAttentionList = []
ExpH2StroopMathStroopHanoistackedFlattenedResumptionList = []

# Hanoi Accuracy
ExpH2StroopMathStroopHanoicollectSumsMovesHanoiAssessment = []
ExpH2StroopMathStroopHanoicollectSumsMovesHanoiTraining = []
ExpH2StroopMathStroopHanoicollectSumsMovesHanoiTesting = []
# Hanoi Speed
ExpH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiAssessment = []
ExpH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiTraining = []
ExpH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ExpH2StroopMathStroopHanoiCollectCorrectnessDrawAssessment = []
ExpH2StroopMathStroopHanoiCollectCorrectnessDrawTraining = []
ExpH2StroopMathStroopHanoiCollectCorrectnessDrawTesting = []
# Draw Speed
ExpH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawAssessment = []
ExpH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawTraining = []
ExpH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawTesting = []

ExpH2StroopMathStroopHanoicollectSumsMovesAndSequencesAssessment = []
ExpH2StroopMathStroopHanoicollectSumsMovesAndSequencesTesting = []

ExpH2StroopMathStroopHanoicollectSumsCompletionTimesAssessment = []
ExpH2StroopMathStroopHanoicollectSumsCompletionTimesTesting = []

ExpH2StroopMathStroopHanoiDiffResumption = []
ExpH2StroopMathStroopHanoiDiffInterruption = []
ExpH2StroopMathStroopHanoiDiffAccuracy = []
ExpH2StroopMathStroopHanoiDiffSpeed = []



# Globally Declared/Localized Variational Use: ExpH2MathStroopMathHanoi
ExpH2MathStroopMathHanoiSumResumptionLagsHanoiAssessment = []
ExpH2MathStroopMathHanoiSumResumptionLagsHanoiTesting = []

ExpH2MathStroopMathHanoiSumInterruptionLagsHanoiAssessment = []
ExpH2MathStroopMathHanoiSumInterruptionLagsHanoiTesting = []

ExpH2MathStroopMathHanoistackedFlattenedAttentionList = []
ExpH2MathStroopMathHanoistackedFlattenedResumptionList = []

# Hanoi Accuracy
ExpH2MathStroopMathHanoicollectSumsMovesHanoiAssessment = []
ExpH2MathStroopMathHanoicollectSumsMovesHanoiTraining = []
ExpH2MathStroopMathHanoicollectSumsMovesHanoiTesting = []
# Hanoi Speed
ExpH2MathStroopMathHanoicollectSumsCompletionTimeHanoiAssessment = []
ExpH2MathStroopMathHanoicollectSumsCompletionTimeHanoiTraining = []
ExpH2MathStroopMathHanoicollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ExpH2MathStroopMathHanoiCollectCorrectnessDrawAssessment = []
ExpH2MathStroopMathHanoiCollectCorrectnessDrawTraining = []
ExpH2MathStroopMathHanoiCollectCorrectnessDrawTesting = []
# Draw Speed
ExpH2MathStroopMathHanoiCollectSumsCompletionTimeDrawAssessment = []
ExpH2MathStroopMathHanoiCollectSumsCompletionTimeDrawTraining = []
ExpH2MathStroopMathHanoiCollectSumsCompletionTimeDrawTesting = []

ExpH2MathStroopMathHanoicollectSumsMovesAndSequencesAssessment = []
ExpH2MathStroopMathHanoicollectSumsMovesAndSequencesTesting = []

ExpH2MathStroopMathHanoicollectSumsCompletionTimesAssessment = []
ExpH2MathStroopMathHanoicollectSumsCompletionTimesTesting = []

ExpH2MathStroopMathHanoiDiffResumption = []
ExpH2MathStroopMathHanoiDiffInterruption = []
ExpH2MathStroopMathHanoiDiffAccuracy = []
ExpH2MathStroopMathHanoiDiffSpeed = []

# Globally Declared/Localized Variational Use: ControlH1DrawHanoiDrawStroop
ControlH1DrawHanoiDrawStroopSumResumptionLagsHanoiAssessment = []
ControlH1DrawHanoiDrawStroopSumResumptionLagsHanoiTesting = []

ControlH1DrawHanoiDrawStroopSumInterruptionLagsHanoiAssessment = []
ControlH1DrawHanoiDrawStroopSumInterruptionLagsHanoiTesting = []

ControlH1DrawHanoiDrawStroopstackedFlattenedAttentionList = []
ControlH1DrawHanoiDrawStroopstackedFlattenedResumptionList = []

# Hanoi Accuracy
ControlH1DrawHanoiDrawStroopcollectSumsMovesHanoiAssessment = []
ControlH1DrawHanoiDrawStroopcollectSumsMovesHanoiTraining = []
ControlH1DrawHanoiDrawStroopcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiAssessment = []
ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiTraining = []
ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ControlH1DrawHanoiDrawStroopCollectCorrectnessDrawAssessment = []
ControlH1DrawHanoiDrawStroopCollectCorrectnessDrawTraining = []
ControlH1DrawHanoiDrawStroopCollectCorrectnessDrawTesting = []
# Draw Speed
ControlH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawAssessment = []
ControlH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawTraining = []
ControlH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawTesting = []

ControlH1DrawHanoiDrawStroopcollectSumsMovesAndSequencesAssessment = []
ControlH1DrawHanoiDrawStroopcollectSumsMovesAndSequencesTesting = []

ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimesAssessment = []
ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimesTesting = []

ControlH1DrawHanoiDrawStroopDiffResumption = []
ControlH1DrawHanoiDrawStroopDiffInterruption = []
ControlH1DrawHanoiDrawStroopDiffAccuracy = []
ControlH1DrawHanoiDrawStroopDiffSpeed = []


# Globally Declared/Localized Variational Use: ControlH1DrawHanoiDrawMath
ControlH1DrawHanoiDrawMathSumResumptionLagsHanoiAssessment = []
ControlH1DrawHanoiDrawMathSumResumptionLagsHanoiTesting = []

ControlH1DrawHanoiDrawMathSumInterruptionLagsHanoiAssessment = []
ControlH1DrawHanoiDrawMathSumInterruptionLagsHanoiTesting = []

ControlH1DrawHanoiDrawMathstackedFlattenedAttentionList = []
ControlH1DrawHanoiDrawMathstackedFlattenedResumptionList = []

# Hanoi Accuracy
ControlH1DrawHanoiDrawMathcollectSumsMovesHanoiAssessment = []
ControlH1DrawHanoiDrawMathcollectSumsMovesHanoiTraining = []
ControlH1DrawHanoiDrawMathcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ControlH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiAssessment = []
ControlH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiTraining = []
ControlH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ControlH1DrawHanoiDrawMathCollectCorrectnessDrawAssessment = []
ControlH1DrawHanoiDrawMathCollectCorrectnessDrawTraining = []
ControlH1DrawHanoiDrawMathCollectCorrectnessDrawTesting = []
# Draw Speed
ControlH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawAssessment = []
ControlH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawTraining = []
ControlH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawTesting = []

ControlH1DrawHanoiDrawMathcollectSumsMovesAndSequencesAssessment = []
ControlH1DrawHanoiDrawMathcollectSumsMovesAndSequencesTesting = []

ControlH1DrawHanoiDrawMathcollectSumsCompletionTimesAssessment = []
ControlH1DrawHanoiDrawMathcollectSumsCompletionTimesTesting = []

ControlH1DrawHanoiDrawMathDiffResumption = []
ControlH1DrawHanoiDrawMathDiffInterruption = []
ControlH1DrawHanoiDrawMathDiffAccuracy = []
ControlH1DrawHanoiDrawMathDiffSpeed = []

# Globally Declared/Localized Variational Use: ControlH1HanoiDrawHanoiStroop
ControlH1HanoiDrawHanoiStroopSumResumptionLagsHanoiAssessment = []
ControlH1HanoiDrawHanoiStroopSumResumptionLagsHanoiTesting = []

ControlH1HanoiDrawHanoiStroopSumInterruptionLagsHanoiAssessment = []
ControlH1HanoiDrawHanoiStroopSumInterruptionLagsHanoiTesting = []

ControlH1HanoiDrawHanoiStroopstackedFlattenedAttentionList = []
ControlH1HanoiDrawHanoiStroopstackedFlattenedResumptionList = []

# Hanoi Accuracy
ControlH1HanoiDrawHanoiStroopcollectSumsMovesHanoiAssessment = []
ControlH1HanoiDrawHanoiStroopcollectSumsMovesHanoiTraining = []
ControlH1HanoiDrawHanoiStroopcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiAssessment = []
ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiTraining = []
ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ControlH1HanoiDrawHanoiStroopCollectCorrectnessDrawAssessment = []
ControlH1HanoiDrawHanoiStroopCollectCorrectnessDrawTraining = []
ControlH1HanoiDrawHanoiStroopCollectCorrectnessDrawTesting = []
# Draw Speed
ControlH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawAssessment = []
ControlH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawTraining = []
ControlH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawTesting = []

ControlH1HanoiDrawHanoiStroopcollectSumsMovesAndSequencesAssessment = []
ControlH1HanoiDrawHanoiStroopcollectSumsMovesAndSequencesTesting = []

ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimesAssessment = []
ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimesTesting = []

ControlH1HanoiDrawHanoiStroopDiffResumption = []
ControlH1HanoiDrawHanoiStroopDiffInterruption = []
ControlH1HanoiDrawHanoiStroopDiffAccuracy = []
ControlH1HanoiDrawHanoiStroopDiffSpeed = []

# Globally Declared/Localized Variational Use: ControlH1HanoiDrawHanoiMath
ControlH1HanoiDrawHanoiMathSumResumptionLagsHanoiAssessment = []
ControlH1HanoiDrawHanoiMathSumResumptionLagsHanoiTesting = []

ControlH1HanoiDrawHanoiMathSumInterruptionLagsHanoiAssessment = []
ControlH1HanoiDrawHanoiMathSumInterruptionLagsHanoiTesting = []

ControlH1HanoiDrawHanoiMathstackedFlattenedAttentionList = []
ControlH1HanoiDrawHanoiMathstackedFlattenedResumptionList = []

# Hanoi Accuracy
ControlH1HanoiDrawHanoiMathcollectSumsMovesHanoiAssessment = []
ControlH1HanoiDrawHanoiMathcollectSumsMovesHanoiTraining = []
ControlH1HanoiDrawHanoiMathcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiAssessment = []
ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiTraining = []
ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ControlH1HanoiDrawHanoiMathCollectCorrectnessDrawAssessment = []
ControlH1HanoiDrawHanoiMathCollectCorrectnessDrawTraining = []
ControlH1HanoiDrawHanoiMathCollectCorrectnessDrawTesting = []
# Draw Speed
ControlH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawAssessment = []
ControlH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawTraining = []
ControlH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawTesting = []

ControlH1HanoiDrawHanoiMathcollectSumsMovesAndSequencesAssessment = []
ControlH1HanoiDrawHanoiMathcollectSumsMovesAndSequencesTesting = []

ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimesAssessment = []
ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimesTesting = []

ControlH1HanoiDrawHanoiMathDiffResumption = []
ControlH1HanoiDrawHanoiMathDiffInterruption = []
ControlH1HanoiDrawHanoiMathDiffAccuracy = []
ControlH1HanoiDrawHanoiMathDiffSpeed = []

# Globally Declared/Localized Variational Use: ControlH2StroopMathStroopDraw
ControlH2StroopMathStroopDrawSumResumptionLagsHanoiAssessment = []
ControlH2StroopMathStroopDrawSumResumptionLagsHanoiTesting = []

ControlH2StroopMathStroopDrawSumInterruptionLagsHanoiAssessment = []
ControlH2StroopMathStroopDrawSumInterruptionLagsHanoiTesting = []

ControlH2StroopMathStroopDrawstackedFlattenedAttentionList = []
ControlH2StroopMathStroopDrawstackedFlattenedResumptionList = []

# Hanoi Accuracy
ControlH2StroopMathStroopDrawcollectSumsMovesHanoiAssessment = []
ControlH2StroopMathStroopDrawcollectSumsMovesHanoiTraining = []
ControlH2StroopMathStroopDrawcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ControlH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiAssessment = []
ControlH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiTraining = []
ControlH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ControlH2StroopMathStroopDrawCollectCorrectnessDrawAssessment = []
ControlH2StroopMathStroopDrawCollectCorrectnessDrawTraining = []
ControlH2StroopMathStroopDrawCollectCorrectnessDrawTesting = []
# Draw Speed
ControlH2StroopMathStroopDrawCollectSumsCompletionTimeDrawAssessment = []
ControlH2StroopMathStroopDrawCollectSumsCompletionTimeDrawTraining = []
ControlH2StroopMathStroopDrawCollectSumsCompletionTimeDrawTesting = []

ControlH2StroopMathStroopDrawcollectSumsMovesAndSequencesAssessment = []
ControlH2StroopMathStroopDrawcollectSumsMovesAndSequencesTesting = []

ControlH2StroopMathStroopDrawcollectSumsCompletionTimesAssessment = []
ControlH2StroopMathStroopDrawcollectSumsCompletionTimesTesting = []

ControlH2StroopMathStroopDrawDiffResumption = []
ControlH2StroopMathStroopDrawDiffInterruption = []
ControlH2StroopMathStroopDrawDiffAccuracy = []
ControlH2StroopMathStroopDrawDiffSpeed = []

# Globally Declared/Localized Variational Use: ControlH2MathStroopMathDraw
ControlH2MathStroopMathDrawSumResumptionLagsHanoiAssessment = []
ControlH2MathStroopMathDrawSumResumptionLagsHanoiTesting = []

ControlH2MathStroopMathDrawSumInterruptionLagsHanoiAssessment = []
ControlH2MathStroopMathDrawSumInterruptionLagsHanoiTesting = []

ControlH2MathStroopMathDrawstackedFlattenedAttentionList = []
ControlH2MathStroopMathDrawstackedFlattenedResumptionList = []

# Hanoi Accuracy
ControlH2MathStroopMathDrawcollectSumsMovesHanoiAssessment = []
ControlH2MathStroopMathDrawcollectSumsMovesHanoiTraining = []
ControlH2MathStroopMathDrawcollectSumsMovesHanoiTesting = []
# Hanoi Speed
ControlH2MathStroopMathDrawcollectSumsCompletionTimeHanoiAssessment = []
ControlH2MathStroopMathDrawcollectSumsCompletionTimeHanoiTraining = []
ControlH2MathStroopMathDrawcollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ControlH2MathStroopMathDrawCollectCorrectnessDrawAssessment = []
ControlH2MathStroopMathDrawCollectCorrectnessDrawTraining = []
ControlH2MathStroopMathDrawCollectCorrectnessDrawTesting = []
# Draw Speed
ControlH2MathStroopMathDrawCollectSumsCompletionTimeDrawAssessment = []
ControlH2MathStroopMathDrawCollectSumsCompletionTimeDrawTraining = []
ControlH2MathStroopMathDrawCollectSumsCompletionTimeDrawTesting = []

ControlH2MathStroopMathDrawcollectSumsMovesAndSequencesAssessment = []
ControlH2MathStroopMathDrawcollectSumsMovesAndSequencesTesting = []

ControlH2MathStroopMathDrawcollectSumsCompletionTimesAssessment = []
ControlH2MathStroopMathDrawcollectSumsCompletionTimesTesting = []

ControlH2MathStroopMathDrawDiffResumption = []
ControlH2MathStroopMathDrawDiffInterruption = []
ControlH2MathStroopMathDrawDiffAccuracy = []
ControlH2MathStroopMathDrawDiffSpeed = []

# Globally Declared/Localized Variational Use: ControlH2StroopMathStroopHanoi
ControlH2StroopMathStroopHanoiSumResumptionLagsHanoiAssessment = []
ControlH2StroopMathStroopHanoiSumResumptionLagsHanoiTesting = []

ControlH2StroopMathStroopHanoiSumInterruptionLagsHanoiAssessment = []
ControlH2StroopMathStroopHanoiSumInterruptionLagsHanoiTesting = []

ControlH2StroopMathStroopHanoistackedFlattenedAttentionList = []
ControlH2StroopMathStroopHanoistackedFlattenedResumptionList = []

# Hanoi Accuracy
ControlH2StroopMathStroopHanoicollectSumsMovesHanoiAssessment = []
ControlH2StroopMathStroopHanoicollectSumsMovesHanoiTraining = []
ControlH2StroopMathStroopHanoicollectSumsMovesHanoiTesting = []
# Hanoi Speed
ControlH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiAssessment = []
ControlH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiTraining = []
ControlH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ControlH2StroopMathStroopHanoiCollectCorrectnessDrawAssessment = []
ControlH2StroopMathStroopHanoiCollectCorrectnessDrawTraining = []
ControlH2StroopMathStroopHanoiCollectCorrectnessDrawTesting = []
# Draw Speed
ControlH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawAssessment = []
ControlH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawTraining = []
ControlH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawTesting = []

ControlH2StroopMathStroopHanoicollectSumsMovesAndSequencesAssessment = []
ControlH2StroopMathStroopHanoicollectSumsMovesAndSequencesTesting = []

ControlH2StroopMathStroopHanoicollectSumsCompletionTimesAssessment = []
ControlH2StroopMathStroopHanoicollectSumsCompletionTimesTesting = []

ControlH2StroopMathStroopHanoiDiffResumption = []
ControlH2StroopMathStroopHanoiDiffInterruption = []
ControlH2StroopMathStroopHanoiDiffAccuracy = []
ControlH2StroopMathStroopHanoiDiffSpeed = []


# Globally Declared/Localized Variational Use: ControlH2MathStroopMathHanoi
ControlH2MathStroopMathHanoiSumResumptionLagsHanoiAssessment = []
ControlH2MathStroopMathHanoiSumResumptionLagsHanoiTesting = []

ControlH2MathStroopMathHanoiSumInterruptionLagsHanoiAssessment = []
ControlH2MathStroopMathHanoiSumInterruptionLagsHanoiTesting = []

ControlH2MathStroopMathHanoistackedFlattenedAttentionList = []
ControlH2MathStroopMathHanoistackedFlattenedResumptionList = []

# Hanoi Accuracy
ControlH2MathStroopMathHanoicollectSumsMovesHanoiAssessment = []
ControlH2MathStroopMathHanoicollectSumsMovesHanoiTraining = []
ControlH2MathStroopMathHanoicollectSumsMovesHanoiTesting = []
# Hanoi Speed
ControlH2MathStroopMathHanoicollectSumsCompletionTimeHanoiAssessment = []
ControlH2MathStroopMathHanoicollectSumsCompletionTimeHanoiTraining = []
ControlH2MathStroopMathHanoicollectSumsCompletionTimeHanoiTesting = []
# Draw Accuracy
ControlH2MathStroopMathHanoiCollectCorrectnessDrawAssessment = []
ControlH2MathStroopMathHanoiCollectCorrectnessDrawTraining = []
ControlH2MathStroopMathHanoiCollectCorrectnessDrawTesting = []
# Draw Speed
ControlH2MathStroopMathHanoiCollectSumsCompletionTimeDrawAssessment = []
ControlH2MathStroopMathHanoiCollectSumsCompletionTimeDrawTraining = []
ControlH2MathStroopMathHanoiCollectSumsCompletionTimeDrawTesting = []

ControlH2MathStroopMathHanoicollectSumsMovesAndSequencesAssessment = []
ControlH2MathStroopMathHanoicollectSumsMovesAndSequencesTesting = []

ControlH2MathStroopMathHanoicollectSumsCompletionTimesAssessment = []
ControlH2MathStroopMathHanoicollectSumsCompletionTimesTesting = []

ControlH2MathStroopMathHanoiDiffResumption = []
ControlH2MathStroopMathHanoiDiffInterruption = []
ControlH2MathStroopMathHanoiDiffAccuracy = []
ControlH2MathStroopMathHanoiDiffSpeed = []


for p in all_participants:
    print(p.p_id)

    # study information
    control_arr.append(p.group)
    id_arr.append(p.p_id)
    conditions_arr.append(p.hypotheses)
    starting_task_arr.append(p.starting_task)
    starting_interruption_arr.append(p.starting_interruption)

    # demographics
    d_age.append(p.survey.demographics.age)
    d_gender.append(p.survey.demographics.gender)
    d_education.append(p.survey.demographics.education)

    # diagnosis
    d_asd.append(p.survey.diagnosis.asd)
    d_colorblind.append(p.survey.diagnosis.color_blind)
    d_hearingimpaired.append(p.survey.diagnosis.hearing_impaired)
    d_adhd.append(p.survey.diagnosis.adhd)
    d_prefernottosay.append(p.survey.diagnosis.prefer_not_to_say)
    d_none.append(p.survey.diagnosis.none)

    # assessment primary task survey results
    a_p_e_task.append(p.survey.effort[0].task)
    a_p_e_effort.append(p.survey.effort[0].effort)
    a_p_e_confidence.append(p.survey.effort[0].confidence)

    # assessment interrupting task survey results
    a_i_e_task.append(p.survey.effort[1].task)
    a_i_e_effort.append(p.survey.effort[1].effort)
    a_i_e_confidence.append(p.survey.effort[1].confidence)

    # training primary task survey results
    tr_p_e_task.append(p.survey.effort[2].task)
    tr_p_e_effort.append(p.survey.effort[2].effort)
    tr_p_e_confidence.append(p.survey.effort[2].confidence)

    # training interrupting task survey results
    tr_i_e_task.append(p.survey.effort[3].task)
    tr_i_e_effort.append(p.survey.effort[3].effort)
    tr_i_e_confidence.append(p.survey.effort[3].confidence)

    # testing primary task survey results
    te_p_e_task.append(p.survey.effort[4].task)
    te_p_e_effort.append(p.survey.effort[4].effort)
    te_p_e_confidence.append(p.survey.effort[4].confidence)

    # testing interrupting task survey results
    te_i_e_task.append(p.survey.effort[5].task)
    te_i_e_effort.append(p.survey.effort[5].effort)
    te_i_e_confidence.append(p.survey.effort[5].confidence)

    if p.assessment_interruption.name == "math":
        mathData=MathData()
        totalTime = mathData.totalTime
        correctResponseCount = 0
        numberOfTasksDuringInterruptions = 0
        durationB4AttentionList = []
        firstMoveDurationAttentionList = []
        for allResponses in p.assessment_interruption.interruption.math_tasks:
            if allResponses.reTasked_during_Assessment == False:
                for eachMove in allResponses.mathResponseList:
                    firstMoveDurationAttentionList.append(float(eachMove.timeSpent))
            if allResponses.reTasked_during_Assessment == True:
                for eachMove in allResponses.mathResponseList:
                    totalTime += float(eachMove.timeSpent)
                    if eachMove.reTasked == 1:
                        numberOfTasksDuringInterruptions += 1
                        durationB4AttentionList.append(float(eachMove.timeSpent))
                        allResponses.reTasked_during_Assessment = False
        for correctResponses in p.assessment_interruption.interruption.math_tasks:
            if correctResponses.correct == True:
                correctResponseCount += 1
            totalTime += float(correctResponses.timeSpent)
        totalNumberOfmathTasks = len(p.assessment_interruption.interruption.math_tasks)

        # average time spent on a given task
        mathData.average_time = totalTime/totalNumberOfmathTasks
        averageTimeMathInterruptions = mathData.average_time
        averageTimeMathInterruptionsListAssess.append(mathData.average_time)

        # percentage of all tasks answered correctly
        percentCorrect = correctResponseCount/totalNumberOfmathTasks

        # record data
        a_i_name.append(p.assessment_interruption.name)
        a_i_count.append(totalNumberOfmathTasks)
        a_i_percentage.append(percentCorrect)
        a_i_time.append(averageTimeMathInterruptions)
        a_i_times.append(averageTimeMathInterruptionsListAssess)
        assessInterruptLagsList = durationB4AttentionList
        # #################################
        # #       NON-MODULAR CODE        #
        # #       IN NEXT 2 LINES         #
        # # Because some participants in  #
        # # the control condition start   #
        # # begin the assessment and      #
        # # testing phases with interruptn#
        # # are not actually interruption #
        # # lags; however, the code below #
        # # prepends first act where      #
        # # participant starts the        #
        # # interruptive tasks as an      #
        # # interruption lag even though  #
        # # the interruptn is not preceded#
        # # by a task that would be       #
        # # interrupted and then resumed. #
        # #################################
        if len(assessInterruptLagsList) == 7:
            assessInterruptLagsList.insert(0, firstMoveDurationAttentionList[0])

        # trim first 2 elements of list then test statistically...
        # del durationB4AttentionList[0:2]

    if p.training_interruption.name == "math":
        mathData = MathData()

        # determine the total time spent and number of correct answers in this phase
        correctResponseCount = 0
        totalTime = mathData.totalTime
        numberOfTasksDuringInterruptions = 0
        durationB4AttentionList = []
        for allResponses in p.training_interruption.interruption.math_tasks:
            if allResponses.reTasked_during_Training == True:
                for eachMove in allResponses.mathResponseList:
                    totalTime += float(eachMove.timeSpent)
                    if eachMove.reTasked == 1:
                        numberOfTasksDuringInterruptions += 1
                        durationB4AttentionList.append(float(eachMove.timeSpent))
                        # durationB4AttentionList.append(float(allResponses.timeSpent))
                        allResponses.reTasked_during_Training = False
        if p.group == 1 and len(durationB4AttentionList) < 8:
            durationB4AttentionList += 8 * [0]
            del durationB4AttentionList[0]

        for correctResponses in p.training_interruption.interruption.math_tasks:
            if correctResponses.correct == True:
                correctResponseCount += 1
            totalTime += float(correctResponses.timeSpent)
        totalNumberOfmathTasks = len(p.training_interruption.interruption.math_tasks)

        # calculate average time per given task
        mathData.average_time = totalTime/totalNumberOfmathTasks
        averageTimeMathInterruptions = mathData.average_time
        averageTimeMathInterruptionsListTrain.append(mathData.average_time)

        # calculate the percentage of tasks answered correctly
        percentCorrect = correctResponseCount / totalNumberOfmathTasks

        # record data
        tr_i_name.append(p.training_interruption.name)
        tr_i_count.append(len(p.training_interruption.interruption.math_tasks))
        tr_i_percentage.append(percentCorrect)
        tr_i_time.append(averageTimeMathInterruptions)
        tr_i_times.append(averageTimeMathInterruptionsListTrain)
        trainingInterruptLagsList = durationB4AttentionList

    if p.testing_interruption.name == "math":
        mathData = MathData()

        # determine the total time spent and number of correct tasks in this phase
        correctResponseCount = 0
        totalTime = mathData.totalTime
        numberOfTasksDuringInterruptions = 0
        durationB4AttentionList = []
        firstMoveDurationAttentionList = []
        for allResponses in p.testing_interruption.interruption.math_tasks:
            if allResponses.reTasked_during_Testing == False:
                for eachMove in allResponses.mathResponseList:
                    firstMoveDurationAttentionList.append(float(eachMove.timeSpent))
            if allResponses.reTasked_during_Testing == True:
                for eachMove in allResponses.mathResponseList:
                    totalTime += float(eachMove.timeSpent)
                    if eachMove.reTasked == 1:
                        numberOfTasksDuringInterruptions +=1
                        durationB4AttentionList.append(float(eachMove.timeSpent))
                        # durationB4AttentionList.append(float(allResponses.timeSpent))
                        allResponses.reTasked_during_Testing = False
        for correctResponses in p.testing_interruption.interruption.math_tasks:
            # durationB4AttentionList.append(float(correctResponses.timeSpent))
            if correctResponses.correct == True:
                correctResponseCount += 1
            totalTime += float(correctResponses.timeSpent)
        totalNumberOfmathTasks = len(p.testing_interruption.interruption.math_tasks)

        # calculate the time spent per given task
        mathData.average_time = totalTime/totalNumberOfmathTasks
        averageTimeMathInterruptions = mathData.average_time
        averageTimeMathInterruptionsListTest.append(mathData.average_time)

        # calculate the percentage of tasks answered correctly
        percentCorrect = correctResponseCount / totalNumberOfmathTasks

        # record data
        te_i_name.append(p.testing_interruption.name)
        te_i_count.append(len(p.testing_interruption.interruption.math_tasks))
        te_i_percentage.append(percentCorrect)
        te_i_time.append(averageTimeMathInterruptions)
        te_i_times.append(averageTimeMathInterruptionsListTest)
        testingInterruptLagsList = durationB4AttentionList
        # #################################
        # #       NON-MODULAR CODE        #
        # #       IN NEXT 2 LINES         #
        # # Because some participants in  #
        # # the control condition start   #
        # # begin the assessment and      #
        # # testing phases with interruptn#
        # # are not actually interruption #
        # # lags; however, the code below #
        # # prepends first act where      #
        # # participant starts the        #
        # # interruptive tasks as an      #
        # # interruption lag even though  #
        # # the interruptn is not preceded#
        # # by a task that would be       #
        # # interrupted and then resumed. #
        # #################################
        if len(testingInterruptLagsList) == 7:
            testingInterruptLagsList.insert(0, firstMoveDurationAttentionList[0])


    if p.assessment_interruption.name == "stroop":
        stroopData=StroopData()

        totalTime = stroopData.totalTime
        correctResponseCount = 0

        numberOfTasksDuringInterruptions = 0
        durationB4AttentionList = []
        firstMoveDurationAttentionList = []
        for allResponses in p.assessment_interruption.interruption.stroop_tasks:
            if allResponses.reTaskedDuringStroopAssessment == True:
                for eachMove in allResponses.stroopResponseList:
                    firstMoveDurationAttentionList.append(float(eachMove.timeSpent))
            if allResponses.reTaskedDuringStroopAssessment == True:
                for eachMove in allResponses.stroopResponseList:
                    totalTime += float(eachMove.timeSpent)
                    if eachMove.reTasked == 1:
                        numberOfTasksDuringInterruptions += 1
                        durationB4AttentionList.append(float(eachMove.timeSpent))
                        allResponses.reTasked_during_Training = False

        for correctResponses in p.assessment_interruption.interruption.stroop_tasks:
            if correctResponses.correct == True:
                correctResponseCount += 1
            totalTime += float(correctResponses.timeSpent)
        totalNumberOfStroopTasks = len(p.assessment_interruption.interruption.stroop_tasks)

        stroopData.average_time = totalTime/totalNumberOfStroopTasks
        averageTimeStroopInterruptions = stroopData.average_time
        averageTimeStroopInterruptionsListAssess.append(stroopData.average_time)

        percentCorrect = correctResponseCount / totalNumberOfStroopTasks

        a_i_name.append(p.assessment_interruption.name)
        a_i_count.append(totalNumberOfStroopTasks)
        a_i_percentage.append(percentCorrect)
        a_i_time.append(averageTimeStroopInterruptions)
        a_i_times.append(averageTimeStroopInterruptionsListAssess)
        assessInterruptLagsList = durationB4AttentionList
        # #################################
        # #       NON-MODULAR CODE        #
        # #       IN NEXT 2 LINES         #
        # # Because some participants in  #
        # # the control condition start   #
        # # begin the assessment and      #
        # # testing phases with interruptn#
        # # are not actually interruption #
        # # lags; however, the code below #
        # # prepends first act where      #
        # # participant starts the        #
        # # interruptive tasks as an      #
        # # interruption lag even though  #
        # # the interruptn is not preceded#
        # # by a task that would be       #
        # # interrupted and then resumed. #
        # #################################
        if len(assessInterruptLagsList) == 7:
            assessInterruptLagsList.insert(0, firstMoveDurationAttentionList[0])
        # the line below was done to test whether dropping the first 2 elements changes the stats
        # it was done in here as well as in the assessment phase of the math interruption
        # trim first 2 elements of list then test statistically...
        # del durationB4AttentionList[0:2]

    if p.training_interruption.name == "stroop":
        stroopData = StroopData()

        totalTime = stroopData.totalTime
        correctResponseCount = 0

        numberOfTasksDuringInterruptions = 0
        durationB4AttentionList = []
        for allResponses in p.training_interruption.interruption.stroop_tasks:
            if allResponses.reTaskedDuringStroopTraining == True:
                for eachMove in allResponses.stroopResponseList:
                    totalTime += float(eachMove.timeSpent)
                    if eachMove.reTasked == 1:
                        numberOfTasksDuringInterruptions += 1
                        durationB4AttentionList.append(float(eachMove.timeSpent))
                        allResponses.reTasked_during_Training = False

        if p.group == 1 and len(durationB4AttentionList) < 8:
            durationB4AttentionList += 8 * [0]
            del durationB4AttentionList[0]

        for correctResponses in p.training_interruption.interruption.stroop_tasks:
            if correctResponses.correct == True:
                correctResponseCount += 1
            totalTime += float(correctResponses.timeSpent)
        totalNumberOfStroopTasks = len(p.training_interruption.interruption.stroop_tasks)

        stroopData.average_time = totalTime/totalNumberOfStroopTasks
        averageTimeStroopInterruptions = stroopData.average_time
        averageTimeStroopInterruptionsListTrain.append(stroopData.average_time)

        percentCorrect = correctResponseCount / totalNumberOfStroopTasks

        tr_i_name.append(p.training_interruption.name)
        tr_i_count.append(totalNumberOfStroopTasks)
        tr_i_percentage.append(percentCorrect)
        tr_i_time.append(averageTimeStroopInterruptions)
        tr_i_times.append(averageTimeStroopInterruptionsListTrain)
        trainingInterruptLagsList = durationB4AttentionList

    if p.testing_interruption.name == "stroop":
        stroopData = StroopData()

        totalTime = stroopData.totalTime
        correctResponseCount = 0

        numberOfTasksDuringInterruptions = 0
        durationB4AttentionList = []
        firstMoveDurationAttentionList = []
        for allResponses in p.testing_interruption.interruption.stroop_tasks:
            if allResponses.reTaskedDuringStroopTesting == False:
                for eachMove in allResponses.stroopResponseList:
                    firstMoveDurationAttentionList.append(float(eachMove.timeSpent))
            if allResponses.reTaskedDuringStroopTesting == True:
                for eachMove in allResponses.stroopResponseList:
                    totalTime += float(eachMove.timeSpent)
                    if eachMove.reTasked == 1:
                        numberOfTasksDuringInterruptions += 1
                        durationB4AttentionList.append(float(eachMove.timeSpent))
                        allResponses.reTasked_during_Training = False

        for correctResponses in p.testing_interruption.interruption.stroop_tasks:
            if correctResponses.correct == True:
                correctResponseCount += 1
            totalTime += float(correctResponses.timeSpent)
        totalNumberOfStroopTasks = len(p.testing_interruption.interruption.stroop_tasks)

        stroopData.average_time = totalTime/totalNumberOfStroopTasks
        averageTimeStroopInterruptions = stroopData.average_time
        averageTimeStroopInterruptionsListTest.append(stroopData.average_time)

        percentCorrect = correctResponseCount / totalNumberOfStroopTasks

        te_i_name.append(p.testing_interruption.name)
        te_i_count.append(totalNumberOfStroopTasks)
        te_i_percentage.append(percentCorrect)
        te_i_time.append(averageTimeStroopInterruptions)
        te_i_times.append(averageTimeStroopInterruptionsListTest)
        testingInterruptLagsList = durationB4AttentionList
        # #################################
        # #       NON-MODULAR CODE        #
        # #       IN NEXT 2 LINES         #
        # # Because some participants in  #
        # # the control condition start   #
        # # begin the assessment and      #
        # # testing phases with interruptn#
        # # are not actually interruption #
        # # lags; however, the code below #
        # # prepends first act where      #
        # # participant starts the        #
        # # interruptive tasks as an      #
        # # interruption lag even though  #
        # # the interruptn is not preceded#
        # # by a task that would be       #
        # # interrupted and then resumed. #
        # #################################
        if len(testingInterruptLagsList) == 7:
            testingInterruptLagsList.insert(0, firstMoveDurationAttentionList[0])


    if p.assessment_task.name == "draw":
        drawTask = DrawTask()
        drawData = DrawData()

        # determine weighted correctness
        totalTimeEntirelyCorrectAssessment = drawTask.time # how much time spent until answering a draw task fully correct
        totalDrawTaskEntirelyCorrectAssessment = 0 # how much time spent to give completely correct answers in this phase
        fiftyPercentCorrect = 0
        twentyFivePercentCorrect = 0
        zeroPercentCorrect = 0
        correctResponsesAssessment = []
        timesCorrespondingToResponseAssessment = []
        checkOnResponsesLength = []
        # weighted by thirds (0, 1/3, 2/3, and 3/3 to correspond to 0 correct, 1, 2, or 3 correct) per Scaz
        # Added "correctResponsesAssessment.append(1 / 3) etc. to 25% and 50%"
        for correctResponses in p.assessment_task.task.draw_tasks:
            checkOnResponsesLength.append(correctResponses.percentage_correct)
            if correctResponses.percentage_correct == "100%":
                totalDrawTaskEntirelyCorrectAssessment +=1
                totalTimeEntirelyCorrectAssessment += float(correctResponses.time)
                correctResponsesAssessment.append(1)
            if correctResponses.percentage_correct == "50%":
                fiftyPercentCorrect +=1
                # correctResponsesAssessment.append(.5)
                totalTimeEntirelyCorrectAssessment += float(correctResponses.time)
                correctResponsesAssessment.append(2 / 3)
            if correctResponses.percentage_correct == "25%":
                twentyFivePercentCorrect +=1
                # correctResponsesAssessment.append(.25)
                totalTimeEntirelyCorrectAssessment += float(correctResponses.time)
                correctResponsesAssessment.append(1 / 3)
            if correctResponses.percentage_correct == "0%":
                zeroPercentCorrect +=1
                totalTimeEntirelyCorrectAssessment += float(correctResponses.time)
                correctResponsesAssessment.append(0)
        totalNumberOfDrawTasks = len(p.assessment_task.task.draw_tasks)
        weightedCorrectness = (totalDrawTaskEntirelyCorrectAssessment*1+fiftyPercentCorrect*.5+twentyFivePercentCorrect*.25 + zeroPercentCorrect * 0)
        drawData.average_correctness = weightedCorrectness/totalNumberOfDrawTasks
        averageEntirelyCorrectAssess = drawTask.percentage_correct = totalDrawTaskEntirelyCorrectAssessment/totalNumberOfDrawTasks

        # time spent to answer correctly
        drawData.averageTimeToAnswerDrawTaskEntirelyCorrect = totalTimeEntirelyCorrectAssessment/totalDrawTaskEntirelyCorrectAssessment
        averageTimeToAnswerDrawTaskEntirelyCorrectAssess = drawData.averageTimeToAnswerDrawTaskEntirelyCorrect
        averageTimeToAnswerDrawTaskEntirelyCorrectListAssess.append(drawData.averageTimeToAnswerDrawTaskEntirelyCorrect)

        # total number of tasks given
        iterant = 0
        totalNumberOfMovesBeforeCompleteForAllDrawTasksPerPhasePerParticipant = 0
        numberOfDrawTasksPerPhasePerParticipant = len(p.assessment_task.task.draw_tasks)

        # total time and resumption
        totalTime = 0
        durationB4resumptionList = []
        numberOfInterruptionsDuringTask = 0
        for eachDrawTask in p.assessment_task.task.draw_tasks:
            if eachDrawTask.interrupted_during_task == True:
                for eachMove in eachDrawTask.draw_response_list:
                    totalTime += float(eachMove.timeSpent)
                    if eachMove.after_interruption == 1:
                        numberOfInterruptionsDuringTask += 1
                        durationB4resumptionList.append(float(eachMove.timeSpent))
                        p.average_time_to_answer_after_interruption = sum(durationB4resumptionList) / len(
                            durationB4resumptionList)
            totalNumberOfMovesBeforeCompleteForAllDrawTasksPerPhasePerParticipant += len(
                p.assessment_task.task.draw_tasks[iterant].draw_response_list)
            timesCorrespondingToResponseAssessment.append(float(eachMove.timeSpent))
            iterant += 1
        averageTimeRespondAfterInterruptionListAssessment.append(p.average_time_to_answer_after_interruption)

        # record data
        a_p_name.append(p.assessment_task.name)
        a_p_count.append(len(p.assessment_task.task.draw_tasks))
        a_p_correctness.append(weightedCorrectness)
        a_p_time.append(averageTimeToAnswerDrawTaskEntirelyCorrectAssess)
        a_p_times.append(averageTimeToAnswerDrawTaskEntirelyCorrectListAssess)
        a_p_percentage.append(drawData.average_correctness)
        a_p_percentage100.append(drawTask.percentage_correct)
        a_p_resumption.append(p.average_time_to_answer_after_interruption)
        a_p_resumptions.append(durationB4resumptionList)
        a_p_interruptions.append("N/A") # consecutive batch of interruptions is not relevant to draw !!!
        a_p_movestotal.append("N/A") # how many moves it takes to complete a draw task is not relevant
        a_p_movetasktime.append(averageTimeRespondAfterInterruptionListAssessment) # the average time after a click
        assessResumptionLagsList = durationB4resumptionList
        # accuracyInAssessmentSum = totalDrawTaskEntirelyCorrectAssessment
        accuracyInAssessmentSum = sum(correctResponsesAssessment) # new method uses weighted accuracy

        accuracyInAssessmentSum /= 16
        accuracyInAssessmentSum = 2 - accuracyInAssessmentSum
        accuracyInAssessmentSum *= 16

        # speedInAssessmentSum = totalTimeEntirelyCorrectAssessment
        speedInAssessmentSum = sum(timesCorrespondingToResponseAssessment)
        speedInAssessmentList = timesCorrespondingToResponseAssessment
        accuracyInAssessmentList = correctResponsesAssessment

    if p.training_task.name == "draw":
        drawTask = DrawTask()
        drawData = DrawData()

        totalTimeEntirelyCorrectTraining = drawTask.time
        totalDrawTaskEntirelyCorrectTraining = 0
        fiftyPercentCorrect = 0
        twentyFivePercentCorrect = 0
        zeroPercentCorrect = 0
        correctResponsesTrain = []
        for correctResponses in p.training_task.task.draw_tasks:
            if correctResponses.percentage_correct == "100%":
                totalDrawTaskEntirelyCorrectTraining +=1
                totalTimeEntirelyCorrectTraining += float(correctResponses.time)
                correctResponsesTrain.append(1)
            if correctResponses.percentage_correct == "50%":
                fiftyPercentCorrect +=1
                # correctResponsesTrain.append(.5)
                correctResponsesTrain.append(2 / 3)
            if correctResponses.percentage_correct == "25%":
                twentyFivePercentCorrect +=1
                # correctResponsesTrain.append(.25)
                correctResponsesTrain.append(1 / 3)
            if correctResponses.percentage_correct == "0%":
                zeroPercentCorrect +=1
                correctResponsesTrain.append(0)
        totalNumberOfDrawTasks = len(p.training_task.task.draw_tasks)
        weightedCorrectness = (totalDrawTaskEntirelyCorrectTraining*1+fiftyPercentCorrect*.5+twentyFivePercentCorrect*.25 + zeroPercentCorrect * 0)
        drawData.average_correctness = weightedCorrectness/totalNumberOfDrawTasks
        averageEntirelyCorrectTraining = drawTask.percentage_correct = totalDrawTaskEntirelyCorrectTraining/totalNumberOfDrawTasks


        drawData.averageTimeToAnswerDrawTaskEntirelyCorrect = totalTimeEntirelyCorrectTraining/totalDrawTaskEntirelyCorrectTraining
        averageTimeToAnswerDrawTaskEntirelyCorrectTraining = drawData.averageTimeToAnswerDrawTaskEntirelyCorrect
        averageTimeToAnswerDrawTaskEntirelyCorrectListTrain.append(drawData.averageTimeToAnswerDrawTaskEntirelyCorrect) # changed from 'average_correctness' to averageTimeToAnswerDrawTaskEntirelyCorrect

        iterant = 0
        totalNumberOfMovesBeforeCompleteForAllDrawTasksPerPhasePerParticipant = 0
        numberOfDrawTasksPerPhasePerParticipant = len(p.training_task.task.draw_tasks)

        totalTime = 0
        durationB4resumptionList = []
        numberOfInterruptionsDuringTask = 0
        for eachDrawTask in p.training_task.task.draw_tasks:
            if eachDrawTask.interrupted_during_task == True:
                for eachMove in eachDrawTask.draw_response_list:
                    totalTime += float(eachMove.timeSpent)
                    if eachMove.after_interruption == 1:
                        numberOfInterruptionsDuringTask += 1
                        durationB4resumptionList.append(float(eachMove.timeSpent))
                        p.average_time_to_answer_after_interruption = sum(durationB4resumptionList) / len(
                            durationB4resumptionList)
            totalNumberOfMovesBeforeCompleteForAllDrawTasksPerPhasePerParticipant += len(
                p.training_task.task.draw_tasks[iterant].draw_response_list)
            iterant += 1
        if p.group == 1 and len(durationB4resumptionList) < 8:
            durationB4resumptionList += 8 * [0]

        if p.group == 0:
            averageTimeRespondAfterInterruptionListTraining.append(p.average_time_to_answer_after_interruption)
            tr_p_resumption.append(p.average_time_to_answer_after_interruption)
        else:
            averageTimeRespondAfterInterruptionListTraining.append("N/A")
            tr_p_resumption.append("N/A")

        tr_p_name.append(p.training_task.name)
        tr_p_count.append(totalNumberOfDrawTasks)
        tr_p_correctness.append(weightedCorrectness)
        tr_p_time.append(averageTimeToAnswerDrawTaskEntirelyCorrectTraining)
        tr_p_times.append(averageTimeToAnswerDrawTaskEntirelyCorrectListTrain)
        tr_p_percentage.append(drawData.average_correctness)
        tr_p_percentage100.append(drawTask.percentage_correct)
        tr_p_resumptions.append(durationB4resumptionList)
        tr_p_interruptions.append("N/A")
        tr_p_movestotal.append("N/A")
        tr_p_movetasktime.append(averageTimeRespondAfterInterruptionListTraining)
        trainingResumptionLagsList = durationB4resumptionList
        # accuracyInTrainingSum = totalDrawTaskEntirelyCorrectTraining
        accuracyInTrainingSum = sum(correctResponsesTrain)
        # A score of 16 is 100% correct for each of the 16 Draw tasks...below is  code for handling the conversion
        # Décimaliser...decreasing for larger values (increasing accuracy)
        accuracyInTrainingSum /= 16
        # The value below should be 2 - ...
        accuracyInTrainingSum = 2 - accuracyInTrainingSum
        # Restore value that's flipped about (inverted/'y-axis')
        accuracyInTrainingSum *= 16
        speedInTrainingSum = totalTimeEntirelyCorrectTraining

    if p.testing_task.name == "draw":
        drawTask = DrawTask()
        drawData = DrawData()

        totalTimeEntirelyCorrectTesting = drawTask.time
        totalDrawTaskEntirelyCorrectTesting = 0
        fiftyPercentCorrect = 0
        twentyFivePercentCorrect = 0
        zeroPercentCorrect = 0
        correctResponsesTesting = []
        timesCorrespondingToResponseTesting = []
        for correctResponses in p.testing_task.task.draw_tasks:
            if correctResponses.percentage_correct == "100%":
                totalDrawTaskEntirelyCorrectTesting += 1
                totalTimeEntirelyCorrectTesting += float(correctResponses.time)
                correctResponsesTesting.append(1)
            if correctResponses.percentage_correct == "50%":
                fiftyPercentCorrect += 1
                # correctResponsesTesting.append(.5)
                correctResponsesTesting.append(2 / 3)
                totalTimeEntirelyCorrectTesting += float(correctResponses.time)
            if correctResponses.percentage_correct == "25%":
                twentyFivePercentCorrect += 1
                # correctResponsesTesting.append(.25)
                correctResponsesTesting.append(1 / 3)
                totalTimeEntirelyCorrectTesting += float(correctResponses.time)
            if correctResponses.percentage_correct == "0%":
                zeroPercentCorrect +=1
                correctResponsesTesting.append(0)
                totalTimeEntirelyCorrectTesting += float(correctResponses.time)
        totalNumberOfDrawTasks = len(p.testing_task.task.draw_tasks)
        weightedCorrectness = (totalDrawTaskEntirelyCorrectTesting * 1 + fiftyPercentCorrect * .5 + twentyFivePercentCorrect * .25 + zeroPercentCorrect * 0)
        drawData.average_correctness = weightedCorrectness / totalNumberOfDrawTasks
        averageEntirelyCorrectTesting = drawTask.percentage_correct = totalDrawTaskEntirelyCorrectTesting / totalNumberOfDrawTasks

        drawData.averageTimeToAnswerDrawTaskEntirelyCorrect = 0
        if (totalDrawTaskEntirelyCorrectTesting != 0):
            drawData.averageTimeToAnswerDrawTaskEntirelyCorrect = totalTimeEntirelyCorrectTesting / totalDrawTaskEntirelyCorrectTesting
        averageTimeToAnswerDrawTaskEntirelyCorrectTesting = drawData.averageTimeToAnswerDrawTaskEntirelyCorrect
        averageTimeToAnswerDrawTaskEntirelyCorrectListTest.append(drawData.averageTimeToAnswerDrawTaskEntirelyCorrect)

        iterant = 0
        totalNumberOfMovesBeforeCompleteForAllDrawTasksPerPhasePerParticipant = 0
        numberOfDrawTasksPerPhasePerParticipant = len(p.testing_task.task.draw_tasks)

        totalTime = 0
        durationB4resumptionList = []
        numberOfInterruptionsDuringTask = 0
        for eachDrawTask in p.testing_task.task.draw_tasks:
            if eachDrawTask.interrupted_during_task == True:
                for eachMove in eachDrawTask.draw_response_list:
                    totalTime += float(eachMove.timeSpent)
                    if eachMove.after_interruption == 1:
                        numberOfInterruptionsDuringTask += 1
                        durationB4resumptionList.append(float(eachMove.timeSpent))
                        p.average_time_to_answer_after_interruption = sum(durationB4resumptionList) / len(
                            durationB4resumptionList)
            totalNumberOfMovesBeforeCompleteForAllDrawTasksPerPhasePerParticipant += len(
                p.testing_task.task.draw_tasks[iterant].draw_response_list)
            timesCorrespondingToResponseTesting.append(float(eachMove.timeSpent))

            iterant += 1

        averageTimeRespondAfterInterruptionListTesting.append(p.average_time_to_answer_after_interruption)

        # #################################
        # #       NON-MODULAR CODE        #
        # #       IN NEXT 2 LINES         #
        # # Because the training phase in #
        # # control condition ends with   #
        # # interruptions, the testing    #
        # # phase begins as if the task   #
        # # was just preceded by an       #
        # # interruption; appends first   #
        # # act where participant starts  #
        # # the tasks as a resumption,    #
        # # whereas it is not resuming.   #
        # # Removal of all first elements #
        # # in control & experiment isn't #
        # # appropriate, neither is       #
        # # selectively doing deletin 1st #
        # #################################
        # del durationB4resumptionList[0]
        if p.group == 1 and len(durationB4resumptionList) >= 9:
            del durationB4resumptionList[0]

        te_p_name.append(p.testing_task.name)
        te_p_count.append(totalNumberOfDrawTasks)
        te_p_correctness.append(weightedCorrectness)
        te_p_time.append(averageTimeToAnswerDrawTaskEntirelyCorrectTesting)
        te_p_times.append(averageTimeToAnswerDrawTaskEntirelyCorrectListTest)
        te_p_percentage.append(drawData.average_correctness)
        te_p_percentage100.append(drawTask.percentage_correct)
        te_p_resumption.append(p.average_time_to_answer_after_interruption)
        te_p_resumptions.append(durationB4resumptionList)
        te_p_interruptions.append("N/A")
        te_p_movestotal.append("N/A")
        te_p_movetasktime.append(averageTimeRespondAfterInterruptionListTesting)
        testingResumptionLagsList = durationB4resumptionList
        # accuracyInTestingSum = totalDrawTaskEntirelyCorrectTesting
        accuracyInTestingSum = sum(correctResponsesTesting)

        accuracyInTestingSum /= 16
        accuracyInTestingSum = 2 - accuracyInTestingSum
        accuracyInTestingSum *= 16

        # speedInTestingSum = totalTimeEntirelyCorrectTesting
        speedInTestingSum = sum(timesCorrespondingToResponseTesting)
        speedInTestingList = timesCorrespondingToResponseTesting
        accuracyInTestingList = correctResponsesTesting

    if p.assessment_task.name == "hanoi":
        iterant = 0
        totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerAssessmentPhasePerParticipant = 0
        numberOfHanoiTasksPerPhasePerParticipant = len(p.assessment_task.task.hanoi_tasks)

        totalTime4Completion4AllHanoiTasksPerAssessmentPhasePerParticipant = 0
        durationB4resumptionList = []
        numberOfMovesToCompleteHanoiAssessment = []
        numberOfInterruptionsDuringTask = 0
        numberOfHanoiCompletionsInPhase = []
        timesCorrespondingToResponseAssessment = []
        timerAdder = 0
        for eachHanoiTask in p.assessment_task.task.hanoi_tasks:
            if eachHanoiTask.completed == False:
                for eachMove in eachHanoiTask.hanoi_move_list:
                    timerAdder += float(eachMove.timeSpent)
            timesCorrespondingToResponseAssessment.append(timerAdder)
            timerAdder = 0
            if eachHanoiTask.interrupted_during_Assessment == True:
                for eachMove in eachHanoiTask.hanoi_move_list:
                    totalTime4Completion4AllHanoiTasksPerAssessmentPhasePerParticipant += float(eachMove.timeSpent)
                    if eachMove.after_interruption == 1:
                        numberOfInterruptionsDuringTask +=1
                        durationB4resumptionList.append(float(eachMove.timeSpent))
                        eachHanoiTask.interrupted_during_Assessment = False
            p.moves_to_complete = len(p.assessment_task.task.hanoi_tasks[iterant].hanoi_move_list)
            numberOfMovesToCompleteHanoiAssessment.append(p.moves_to_complete)
            totalNumberOfMovesBeforeCompletePerTask = p.moves_to_complete
            numberOfHanoiCompletionsInPhase.append(totalNumberOfMovesBeforeCompletePerTask)
            totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerAssessmentPhasePerParticipant += \
                len(p.assessment_task.task.hanoi_tasks[iterant].hanoi_move_list)
            iterant+=1
        avgNumberOfMovesBeforeCompleteForAllHanoiTasksPerAssessmentPhasePerParticipant = \
            p.average_moves_to_complete = \
            totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerAssessmentPhasePerParticipant / \
            numberOfHanoiTasksPerPhasePerParticipant
        averageTime4Completion4AllHanoiTasksPerAssessmentPhasePerParticipant = p.average_time_to_complete = totalTime4Completion4AllHanoiTasksPerAssessmentPhasePerParticipant/numberOfHanoiTasksPerPhasePerParticipant
        p.average_time_move_after_interruption = sum(durationB4resumptionList)/len(durationB4resumptionList)
        avgTimesToCompletionForAllHanoiTasksListAssessment.append(p.average_time_to_complete)

        averageTimeMoveAfterInterruptionListAssessment.append(p.average_time_move_after_interruption)
        averageNumberOfMovesBeforeCompleteForAllHanoiTasksListTrain.append(p.average_moves_to_complete)

        a_p_name.append(p.assessment_task.name)
        a_p_count.append(len(p.assessment_task.task.hanoi_tasks))
        a_p_correctness.append("N/A") # hanoi doesn't have a correctness
        a_p_time.append(totalTime4Completion4AllHanoiTasksPerAssessmentPhasePerParticipant) # total time on phase
        a_p_times.append(p.average_time_to_complete) # time per hanoi question
        a_p_percentage.append("N/A") # hanoi doesn't have a correctness
        a_p_percentage100.append("N/A") # hanoi doesn't have a correctness
        a_p_resumption.append(p.average_time_move_after_interruption)
        a_p_resumptions.append(durationB4resumptionList)
        a_p_interruptions.append(numberOfInterruptionsDuringTask)
        a_p_movestotal.append(p.average_moves_to_complete)
        a_p_movetasktime.append(averageTimeMoveAfterInterruptionListAssessment)
        if      (p.group == 0 and p.hypotheses == 1 and p.starting_task == 2) or \
                (p.group == 0 and p.hypotheses == 1 and p.starting_task == 2) or \
                (p.group == 0 and p.hypotheses == 2 and p.starting_task == 2) or \
                (p.group == 0 and p.hypotheses == 2 and p.starting_task == 2) or \
                (p.group == 1 and p.hypotheses == 1 and p.starting_task == 2) or \
                (p.group == 1 and p.hypotheses == 1 and p.starting_task == 2) or \
                (p.group == 1 and p.hypotheses == 2 and p.starting_task == 2) or \
                (p.group == 1 and p.hypotheses == 2 and p.starting_task == 2):
            numberOfMovesToCompleteHanoiStackedAssessment.append(numberOfMovesToCompleteHanoiAssessment)


        assessResumptionLagsList = durationB4resumptionList
        speedInAssessmentSum = sum(timesCorrespondingToResponseAssessment)
        speedInAssessmentList = timesCorrespondingToResponseAssessment

        # fewestHanoiMovesPerTaskAssessmentHardcoded = [1, 7, 7, 14, 13, 13, 7, 7, 3, 3, 14, 13, 3, 3, 7, 3]
        accuracyInAssessmentSum = [n / b for n, b in
                                   zip(numberOfMovesToCompleteHanoiAssessment, fewestHanoiMovesPerTaskAssessmentHardcoded)]
        accuracyInAssessmentSum = sum(accuracyInAssessmentSum)
        accuracyInAssessmentList = numberOfMovesToCompleteHanoiAssessment


    if p.training_task.name == "hanoi":
        iterant = 0
        totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerTrainingPhasePerParticipant = 0
        numberOfHanoiTasksPerPhasePerParticipant = len(p.training_task.task.hanoi_tasks)

        totalTime4Completion4AllHanoiTasksPerTrainingPhasePerParticipant = 0
        durationB4resumptionList = []
        numberOfInterruptionsDuringTask = 0
        numberOfHanoiCompletionsInPhase = []
        for eachHanoiTask in p.training_task.task.hanoi_tasks:
            if eachHanoiTask.interrupted_during_Training == True:
                for eachMove in eachHanoiTask.hanoi_move_list:
                    totalTime4Completion4AllHanoiTasksPerTrainingPhasePerParticipant += float(eachMove.timeSpent)
                    if eachMove.after_interruption == 1:
                        numberOfInterruptionsDuringTask +=1
                        durationB4resumptionList.append(float(eachMove.timeSpent))
                        eachHanoiTask.interrupted_during_Training = False
            p.moves_to_complete = len(p.training_task.task.hanoi_tasks[iterant].hanoi_move_list)
            totalNumberOfMovesBeforeCompletePerTask = p.moves_to_complete
            numberOfHanoiCompletionsInPhase.append(totalNumberOfMovesBeforeCompletePerTask)
            totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerTrainingPhasePerParticipant += \
                len(p.training_task.task.hanoi_tasks[iterant].hanoi_move_list)
            iterant+=1
        if p.group == 1 and len(durationB4resumptionList) < 8:
            durationB4resumptionList += 8 * [0]

        avgNumberOfMovesBeforeCompleteForAllHanoiTasksPerTrainingPhasePerParticipant = \
            p.average_moves_to_complete = \
            totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerTrainingPhasePerParticipant / \
            numberOfHanoiTasksPerPhasePerParticipant
        p.average_moves_to_complete = totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerTrainingPhasePerParticipant/numberOfHanoiTasksPerPhasePerParticipant
        averageTime4Completion4AllHanoiTasksPerTrainingPhasePerParticipant = p.average_time_to_complete = totalTime4Completion4AllHanoiTasksPerTrainingPhasePerParticipant/numberOfHanoiTasksPerPhasePerParticipant

        avgTimesToCompletionForAllHanoiTasksListTraining.append(p.average_time_to_complete) #p.average_time_to_complete.append(totalTime)
        averageNumberOfMovesBeforeCompleteForAllHanoiTasksListTrain.append(p.average_moves_to_complete)

        tr_p_name.append(p.training_task.name)
        tr_p_count.append(len(p.training_task.task.hanoi_tasks))
        tr_p_correctness.append("N/A") # hanoi doesn't have a correctness
        tr_p_time.append(totalTime4Completion4AllHanoiTasksPerTrainingPhasePerParticipant) # total time on phase
        tr_p_times.append(p.average_time_to_complete) # time per hanoi question
        tr_p_percentage.append("N/A") # hanoi doesn't have a correctness
        tr_p_percentage100.append("N/A") # hanoi doesn't have a correctness
        tr_p_resumptions.append(durationB4resumptionList)
        tr_p_interruptions.append(numberOfInterruptionsDuringTask)
        tr_p_movestotal.append(p.average_moves_to_complete)
        tr_p_movetasktime.append(averageTimeMoveAfterInterruptionListTraining)

        # no interruptions are experienced during the training phase of the control
        if p.group == 0:
            p.average_time_move_after_interruption = sum(durationB4resumptionList)/len(durationB4resumptionList)
            averageTimeMoveAfterInterruptionListTraining.append(p.average_time_move_after_interruption)
            tr_p_resumption.append(p.average_time_move_after_interruption)
        else:
            averageTimeMoveAfterInterruptionListTraining.append("N/A")
            tr_p_resumption.append("N/A")

        trainingResumptionLagsList = durationB4resumptionList
        accuracyInTrainingSum = totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerTrainingPhasePerParticipant
        speedInTrainingSum = totalTime4Completion4AllHanoiTasksPerTrainingPhasePerParticipant

    if p.testing_task.name == "hanoi":
        iterant = 0
        totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerTestingPhasePerParticipant = 0
        numberOfHanoiTasksPerPhasePerParticipant = len(p.testing_task.task.hanoi_tasks)

        totalTime4Completion4AllHanoiTasksPerTestingPhasePerParticipant = 0
        durationB4resumptionList = []
        numberOfInterruptionsDuringTask = 0
        numberOfHanoiCompletionsInPhase = []
        numberOfMovesToCompleteHanoiTesting = []
        timesCorrespondingToResponseTesting = []
        timerAdder = 0
        for eachHanoiTask in p.testing_task.task.hanoi_tasks:
            if eachHanoiTask.completed == False:
                for eachMove in eachHanoiTask.hanoi_move_list:
                    timerAdder += float(eachMove.timeSpent)
            timesCorrespondingToResponseTesting.append(timerAdder)

            timerAdder = 0
            if eachHanoiTask.interrupted_during_Testing == True:
                for eachMove in eachHanoiTask.hanoi_move_list:
                    totalTime4Completion4AllHanoiTasksPerTestingPhasePerParticipant += float(eachMove.timeSpent)
                    if eachMove.after_interruption == 1:
                        numberOfInterruptionsDuringTask +=1
                        durationB4resumptionList.append(float(eachMove.timeSpent))
                        eachHanoiTask.interrupted_during_Testing = False
            p.moves_to_complete = len(p.testing_task.task.hanoi_tasks[iterant].hanoi_move_list)
            totalNumberOfMovesBeforeCompletePerTask = p.moves_to_complete
            numberOfMovesToCompleteHanoiTesting.append(totalNumberOfMovesBeforeCompletePerTask)
            numberOfHanoiCompletionsInPhase.append(totalNumberOfMovesBeforeCompletePerTask)
            totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerTestingPhasePerParticipant += \
                len(p.testing_task.task.hanoi_tasks[iterant].hanoi_move_list)
            iterant+=1

        # #################################
        # #       NON-MODULAR CODE        #
        # #       IN NEXT 2 LINES         #
        # # Because the training phase in #
        # # control condition ends with   #
        # # interruptions, the testing    #
        # # phase begins as if the task   #
        # # was just preceded by an       #
        # # interruption; appends first   #
        # # act where participant starts  #
        # # the tasks as a resumption,    #
        # # whereas it is not resuming.   #
        # # Removal of all first elements #
        # # in control & experiment isn't #
        # # appropriate, neither is       #
        # # selectively doing deletin 1st #
        # #################################
        # del durationB4resumptionList[0]
        if p.group == 1 and len(durationB4resumptionList) >= 9:
            del durationB4resumptionList[0]

        avgNumberOfMovesBeforeCompleteForAllHanoiTasksPerTestingPhasePerParticipant = \
            p.average_moves_to_complete = \
            totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerTestingPhasePerParticipant / \
            numberOfHanoiTasksPerPhasePerParticipant
        p.average_moves_to_complete = totalNumberOfMovesBeforeCompleteForAllHanoiTasksPerTestingPhasePerParticipant/numberOfHanoiTasksPerPhasePerParticipant
        averageTime4Completion4AllHanoiTasksPerTestingPhasePerParticipant = p.average_time_to_complete = totalTime4Completion4AllHanoiTasksPerTestingPhasePerParticipant/numberOfHanoiTasksPerPhasePerParticipant
        p.average_time_move_after_interruption = sum(durationB4resumptionList)/len(durationB4resumptionList)
        avgTimesToCompletionForAllHanoiTasksListTesting.append(p.average_time_to_complete)
        averageTimeMoveAfterInterruptionListTesting.append(p.average_time_move_after_interruption)
        averageNumberOfMovesBeforeCompleteForAllHanoiTasksListTest.append(p.average_moves_to_complete)

        te_p_name.append(p.testing_task.name)
        te_p_count.append(len(p.testing_task.task.hanoi_tasks))
        te_p_correctness.append("N/A") # hanoi doesn't have a correctness
        te_p_time.append(totalTime4Completion4AllHanoiTasksPerTestingPhasePerParticipant)
        te_p_times.append(p.average_time_to_complete)
        te_p_percentage.append("N/A") # hanoi doesn't have a correctness
        te_p_percentage100.append("N/A") # hanoi doesn't have a correctness
        te_p_resumption.append(p.average_time_move_after_interruption)
        te_p_resumptions.append(durationB4resumptionList)
        te_p_interruptions.append(numberOfInterruptionsDuringTask)
        te_p_movestotal.append(p.average_moves_to_complete)
        te_p_movetasktime.append(averageTimeMoveAfterInterruptionListTesting)

        if      (p.group == 0 and p.hypotheses == 1 and p.starting_task == 2) or \
                (p.group == 0 and p.hypotheses == 1 and p.starting_task == 2) or \
                (p.group == 0 and p.hypotheses == 2 and p.starting_task == 2) or \
                (p.group == 0 and p.hypotheses == 2 and p.starting_task == 2) or \
                (p.group == 1 and p.hypotheses == 1 and p.starting_task == 2) or \
                (p.group == 1 and p.hypotheses == 1 and p.starting_task == 2) or \
                (p.group == 1 and p.hypotheses == 2 and p.starting_task == 2) or \
                (p.group == 1 and p.hypotheses == 2 and p.starting_task == 2):
            numberOfMovesToCompleteHanoiStackedTesting.append(numberOfMovesToCompleteHanoiTesting)

        testingResumptionLagsList = durationB4resumptionList
        speedInTestingSum = sum(timesCorrespondingToResponseTesting)
        speedInTestingList = timesCorrespondingToResponseTesting

        # fewestHanoiMovesPerTaskTestingHardcoded = [3, 7, 8, 3, 7, 7, 9, 2, 2, 7, 10, 7, 7, 3, 7, 7]
        accuracyInTestingSum = [n / b for n, b in
                                zip(numberOfMovesToCompleteHanoiTesting, fewestHanoiMovesPerTaskTestingHardcoded)]
        accuracyInTestingSum = sum(accuracyInTestingSum)
        accuracyInTestingList = numberOfMovesToCompleteHanoiTesting



    # for loop to stack the required lists by conditions, hypotheses, task/interruption toggling
    # # Experimental Group Demarcation -------------------------------------------------------------------
    if p.group == 0 and p.hypotheses == 1 and p.starting_task == 1 and p.starting_interruption == 1:
        # Experimental group = 0, hypothesis 1 (Task-Toggling) = p.hypotheses = 1
        print("ExpH1DrawHanoiDrawStroop")
        # Calling sortStackAverageStatisticize method
        ExpH1DrawHanoiDrawStroopAttentions, \
        ExpH1DrawHanoiDrawStroopResumptions, \
        ExpH1DrawHanoiDrawStroopAccuracies, \
        ExpH1DrawHanoiDrawStroopSpeeds, \
        ExpH1DrawHanoiDrawStroopResumptionStats, \
        ExpH1DrawHanoiDrawStroopInterruptionStats, \
        ExpH1DrawHanoiDrawStroopAccuracyStats, \
        ExpH1DrawHanoiDrawStroopSpeedStats, \
        ExpH1DrawHanoiDrawStroopDiffResumption, \
        ExpH1DrawHanoiDrawStroopDiffInterruption, \
        ExpH1DrawHanoiDrawStroopDiffAccuracy, \
        ExpH1DrawHanoiDrawStroopDiffSpeed, \
        ExpH1DrawHanoiDrawStroopStackedAttentionList, \
        ExpH1DrawHanoiDrawStroopStackedResumptionList, \
        ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment, \
        ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting, \
        ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment, \
        ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting, \
        ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment, \
        ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting, \
        ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment, \
        ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ExpH1DrawHanoiDrawStroopSumResumptionLagsHanoiAssessment,
                                         ExpH1DrawHanoiDrawStroopSumResumptionLagsHanoiTesting,
                                         ExpH1DrawHanoiDrawStroopSumInterruptionLagsHanoiAssessment,
                                         ExpH1DrawHanoiDrawStroopSumInterruptionLagsHanoiTesting,
                                         ExpH1DrawHanoiDrawStroopstackedFlattenedAttentionList,
                                         ExpH1DrawHanoiDrawStroopstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ExpH1DrawHanoiDrawStroopcollectSumsMovesAndSequencesAssessment,
                                         ExpH1DrawHanoiDrawStroopcollectSumsMovesAndSequencesTesting,
                                         ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimesAssessment,
                                         ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ExpH1DrawHanoiDrawStroop_HanoiAccuracy, \
        ExpH1DrawHanoiDrawStroop_HanoiSpeed, \
        ExpH1DrawHanoiDrawStroop_DrawAccuracy, \
        ExpH1DrawHanoiDrawStroop_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ExpH1DrawHanoiDrawStroopcollectSumsMovesHanoiAssessment,
                ExpH1DrawHanoiDrawStroopcollectSumsMovesHanoiTraining,
                ExpH1DrawHanoiDrawStroopcollectSumsMovesHanoiTesting,
                ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiAssessment,
                ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiTraining,
                ExpH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiTesting,
                ExpH1DrawHanoiDrawStroopCollectCorrectnessDrawAssessment,
                ExpH1DrawHanoiDrawStroopCollectCorrectnessDrawTraining,
                ExpH1DrawHanoiDrawStroopCollectCorrectnessDrawTesting,
                ExpH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawAssessment,
                ExpH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawTraining,
                ExpH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawTesting)

        del ExpH1DrawHanoiDrawStroop_HanoiAccuracy[1]
        del ExpH1DrawHanoiDrawStroop_HanoiSpeed[1]
        del ExpH1DrawHanoiDrawStroop_DrawAccuracy[1]
        del ExpH1DrawHanoiDrawStroop_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH1DrawHanoiDrawStroop_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ExpH1DrawHanoiDrawStroopAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Experimental Group: H1: Path Tracing-Hanoi-Path Tracing Stroop'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ExpH1DrawHanoiDrawStroopAttentions, ExpH1DrawHanoiDrawStroopAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH1DrawHanoiDrawStroop_HanoiAccuracy,
                ExpH1DrawHanoiDrawStroop_HanoiSpeed, ExpH1DrawHanoiDrawStroop_DrawAccuracy, \
                ExpH1DrawHanoiDrawStroop_DrawSpeed, ExpH1DrawHanoiDrawStroopInterruptionStats,
                ExpH1DrawHanoiDrawStroopAccuracyStats, ExpH1DrawHanoiDrawStroopSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH1DrawHanoiDrawStroop_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ExpH1DrawHanoiDrawStroopResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Experimental Group: H1: Path Tracing-Hanoi-Path Tracing Stroop'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ExpH1DrawHanoiDrawStroopResumptions, ExpH1DrawHanoiDrawStroopResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH1DrawHanoiDrawStroop_HanoiAccuracy,
                ExpH1DrawHanoiDrawStroop_HanoiSpeed, ExpH1DrawHanoiDrawStroop_DrawAccuracy, \
                ExpH1DrawHanoiDrawStroop_DrawSpeed, ExpH1DrawHanoiDrawStroopResumptionStats,
                ExpH1DrawHanoiDrawStroopAccuracyStats, ExpH1DrawHanoiDrawStroopSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ExpH1DrawHanoiDrawStroopResumptionLag"
        # ExpH1DrawHanoiDrawStroopResumptionStatsDF = pd.DataFrame(ExpH1DrawHanoiDrawStroopResumptionStats)
        # ExpH1DrawHanoiDrawStroopResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1DrawHanoiDrawStroopResumptionStats.insert(0, "Name", "ExpH1DrawHanoiDrawStroopResumptionLag")

        # filenameForStats = "ExpH1DrawHanoiDrawStroopInterruptionLag"
        # ExpH1DrawHanoiDrawStroopInterruptionStatsDF = pd.DataFrame(ExpH1DrawHanoiDrawStroopInterruptionStats)
        # ExpH1DrawHanoiDrawStroopInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1DrawHanoiDrawStroopInterruptionStats.insert(0, "Name", "ExpH1DrawHanoiDrawStroopInterruptionLag")

        # filenameForStats = "ExpH1DrawHanoiDrawStroopAccuracy"
        # ExpH1DrawHanoiDrawStroopAccuracyStatsDF = pd.DataFrame(ExpH1DrawHanoiDrawStroopAccuracyStats)
        # ExpH1DrawHanoiDrawStroopAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1DrawHanoiDrawStroopAccuracyStats.insert(0, "Name", "ExpH1DrawHanoiDrawStroopAccuracy")

        # filenameForStats = "ExpH1DrawHanoiDrawStroopSpeed"
        # ExpH1DrawHanoiDrawStroopSpeedStatsDF = pd.DataFrame(ExpH1DrawHanoiDrawStroopSpeedStats)
        # ExpH1DrawHanoiDrawStroopSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1DrawHanoiDrawStroopSpeedStats.insert(0, "Name", "ExpH1DrawHanoiDrawStroopSpeed")

    if p.group == 0 and p.hypotheses == 1 and p.starting_task == 1 and p.starting_interruption == 2:
        print("ExpH1DrawHanoiDrawMath")
        # Calling sortStackAverageStatisticize method
        ExpH1DrawHanoiDrawMathAttentions, \
        ExpH1DrawHanoiDrawMathResumptions, \
        ExpH1DrawHanoiDrawMathAccuracies, \
        ExpH1DrawHanoiDrawMathSpeeds, \
        ExpH1DrawHanoiDrawMathResumptionStats, \
        ExpH1DrawHanoiDrawMathInterruptionStats, \
        ExpH1DrawHanoiDrawMathAccuracyStats, \
        ExpH1DrawHanoiDrawMathSpeedStats, \
        ExpH1DrawHanoiDrawMathDiffResumption, \
        ExpH1DrawHanoiDrawMathDiffInterruption, \
        ExpH1DrawHanoiDrawMathDiffAccuracy, \
        ExpH1DrawHanoiDrawMathDiffSpeed, \
        ExpH1DrawHanoiDrawMathStackedAttentionList, \
        ExpH1DrawHanoiDrawMathStackedResumptionList, \
        ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment, \
        ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting, \
        ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment, \
        ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting, \
        ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment, \
        ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting, \
        ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment, \
        ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ExpH1DrawHanoiDrawMathSumResumptionLagsHanoiAssessment,
                                         ExpH1DrawHanoiDrawMathSumResumptionLagsHanoiTesting,
                                         ExpH1DrawHanoiDrawMathSumInterruptionLagsHanoiAssessment,
                                         ExpH1DrawHanoiDrawMathSumInterruptionLagsHanoiTesting,
                                         ExpH1DrawHanoiDrawMathstackedFlattenedAttentionList,
                                         ExpH1DrawHanoiDrawMathstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ExpH1DrawHanoiDrawMathcollectSumsMovesAndSequencesAssessment,
                                         ExpH1DrawHanoiDrawMathcollectSumsMovesAndSequencesTesting,
                                         ExpH1DrawHanoiDrawMathcollectSumsCompletionTimesAssessment,
                                         ExpH1DrawHanoiDrawMathcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ExpH1DrawHanoiDrawMath_HanoiAccuracy, \
        ExpH1DrawHanoiDrawMath_HanoiSpeed, \
        ExpH1DrawHanoiDrawMath_DrawAccuracy, \
        ExpH1DrawHanoiDrawMath_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ExpH1DrawHanoiDrawMathcollectSumsMovesHanoiAssessment,
                ExpH1DrawHanoiDrawMathcollectSumsMovesHanoiTraining,
                ExpH1DrawHanoiDrawMathcollectSumsMovesHanoiTesting,
                ExpH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiAssessment,
                ExpH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiTraining,
                ExpH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiTesting,
                ExpH1DrawHanoiDrawMathCollectCorrectnessDrawAssessment,
                ExpH1DrawHanoiDrawMathCollectCorrectnessDrawTraining,
                ExpH1DrawHanoiDrawMathCollectCorrectnessDrawTesting,
                ExpH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawAssessment,
                ExpH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawTraining,
                ExpH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawTesting)

        del ExpH1DrawHanoiDrawMath_HanoiAccuracy[1]
        del ExpH1DrawHanoiDrawMath_HanoiSpeed[1]
        del ExpH1DrawHanoiDrawMath_DrawAccuracy[1]
        del ExpH1DrawHanoiDrawMath_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH1DrawHanoiDrawMath_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ExpH1DrawHanoiDrawMathAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Experimental Group: H1: Path Tracing-Hanoi-Path Tracing Math'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ExpH1DrawHanoiDrawMathAttentions, ExpH1DrawHanoiDrawMathAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH1DrawHanoiDrawMath_HanoiAccuracy,
                ExpH1DrawHanoiDrawMath_HanoiSpeed, ExpH1DrawHanoiDrawMath_DrawAccuracy, \
                ExpH1DrawHanoiDrawMath_DrawSpeed, ExpH1DrawHanoiDrawMathInterruptionStats,
                ExpH1DrawHanoiDrawMathAccuracyStats, ExpH1DrawHanoiDrawMathSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH1DrawHanoiDrawMath_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ExpH1DrawHanoiDrawMathResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Experimental Group: H1: Path Tracing-Hanoi-Path Tracing Math'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ExpH1DrawHanoiDrawMathResumptions, ExpH1DrawHanoiDrawMathResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH1DrawHanoiDrawMath_HanoiAccuracy,
                ExpH1DrawHanoiDrawMath_HanoiSpeed, ExpH1DrawHanoiDrawMath_DrawAccuracy, \
                ExpH1DrawHanoiDrawMath_DrawSpeed, ExpH1DrawHanoiDrawMathResumptionStats,
                ExpH1DrawHanoiDrawMathAccuracyStats, ExpH1DrawHanoiDrawMathSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ExpH1DrawHanoiDrawMathResumptionLag"
        # ExpH1DrawHanoiDrawMathResumptionStatsDF = pd.DataFrame(ExpH1DrawHanoiDrawMathResumptionStats)
        # ExpH1DrawHanoiDrawMathResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1DrawHanoiDrawMathResumptionStats.insert(0, "Name", "ExpH1DrawHanoiDrawMathResumptionLag")

        # filenameForStats = "ExpH1DrawHanoiDrawMathInterruptionLag"
        # ExpH1DrawHanoiDrawMathInterruptionStatsDF = pd.DataFrame(ExpH1DrawHanoiDrawMathInterruptionStats)
        # ExpH1DrawHanoiDrawMathInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1DrawHanoiDrawMathInterruptionStats.insert(0, "Name", "ExpH1DrawHanoiDrawMathInterruptionLag")

        # filenameForStats = "ExpH1DrawHanoiDrawMathAccuracy"
        # ExpH1DrawHanoiDrawMathAccuracyStatsDF = pd.DataFrame(ExpH1DrawHanoiDrawMathAccuracyStats)
        # ExpH1DrawHanoiDrawMathAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1DrawHanoiDrawMathAccuracyStats.insert(0, "Name", "ExpH1DrawHanoiDrawMathAccuracy")

        # filenameForStats = "ExpH1DrawHanoiDrawMathSpeed"
        # ExpH1DrawHanoiDrawMathSpeedStatsDF = pd.DataFrame(ExpH1DrawHanoiDrawMathSpeedStats)
        # ExpH1DrawHanoiDrawMathSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1DrawHanoiDrawMathSpeedStats.insert(0, "Name", "ExpH1DrawHanoiDrawMathSpeed")

    if p.group == 0 and p.hypotheses == 1 and p.starting_task == 2 and p.starting_interruption == 1:
        print("ExpH1HanoiDrawHanoiStroop")
        # Calling sortStackAverageStatisticize method
        ExpH1HanoiDrawHanoiStroopAttentions, \
        ExpH1HanoiDrawHanoiStroopResumptions, \
        ExpH1HanoiDrawHanoiStroopAccuracies, \
        ExpH1HanoiDrawHanoiStroopSpeeds, \
        ExpH1HanoiDrawHanoiStroopResumptionStats, \
        ExpH1HanoiDrawHanoiStroopInterruptionStats, \
        ExpH1HanoiDrawHanoiStroopAccuracyStats, \
        ExpH1HanoiDrawHanoiStroopSpeedStats, \
        ExpH1HanoiDrawHanoiStroopDiffResumption, \
        ExpH1HanoiDrawHanoiStroopDiffInterruption, \
        ExpH1HanoiDrawHanoiStroopDiffAccuracy, \
        ExpH1HanoiDrawHanoiStroopDiffSpeed, \
        ExpH1HanoiDrawHanoiStroopStackedAttentionList, \
        ExpH1HanoiDrawHanoiStroopStackedResumptionList, \
        ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment, \
        ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting, \
        ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment, \
        ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting, \
        ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment, \
        ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting, \
        ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment, \
        ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ExpH1HanoiDrawHanoiStroopSumResumptionLagsHanoiAssessment,
                                         ExpH1HanoiDrawHanoiStroopSumResumptionLagsHanoiTesting,
                                         ExpH1HanoiDrawHanoiStroopSumInterruptionLagsHanoiAssessment,
                                         ExpH1HanoiDrawHanoiStroopSumInterruptionLagsHanoiTesting,
                                         ExpH1HanoiDrawHanoiStroopstackedFlattenedAttentionList,
                                         ExpH1HanoiDrawHanoiStroopstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ExpH1HanoiDrawHanoiStroopcollectSumsMovesAndSequencesAssessment,
                                         ExpH1HanoiDrawHanoiStroopcollectSumsMovesAndSequencesTesting,
                                         ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimesAssessment,
                                         ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ExpH1HanoiDrawHanoiStroop_HanoiAccuracy, \
        ExpH1HanoiDrawHanoiStroop_HanoiSpeed, \
        ExpH1HanoiDrawHanoiStroop_DrawAccuracy, \
        ExpH1HanoiDrawHanoiStroop_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ExpH1HanoiDrawHanoiStroopcollectSumsMovesHanoiAssessment,
                ExpH1HanoiDrawHanoiStroopcollectSumsMovesHanoiTraining,
                ExpH1HanoiDrawHanoiStroopcollectSumsMovesHanoiTesting,
                ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiAssessment,
                ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiTraining,
                ExpH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiTesting,
                ExpH1HanoiDrawHanoiStroopCollectCorrectnessDrawAssessment,
                ExpH1HanoiDrawHanoiStroopCollectCorrectnessDrawTraining,
                ExpH1HanoiDrawHanoiStroopCollectCorrectnessDrawTesting,
                ExpH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawAssessment,
                ExpH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawTraining,
                ExpH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawTesting)

        del ExpH1HanoiDrawHanoiStroop_HanoiAccuracy[1]
        del ExpH1HanoiDrawHanoiStroop_HanoiSpeed[1]
        del ExpH1HanoiDrawHanoiStroop_DrawAccuracy[1]
        del ExpH1HanoiDrawHanoiStroop_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH1HanoiDrawHanoiStroop_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ExpH1HanoiDrawHanoiStroopAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Experimental Group: H1: Hanoi-Path Tracing-Hanoi Stroop'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ExpH1HanoiDrawHanoiStroopAttentions, ExpH1HanoiDrawHanoiStroopAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH1HanoiDrawHanoiStroop_HanoiAccuracy,
                ExpH1HanoiDrawHanoiStroop_HanoiSpeed, ExpH1HanoiDrawHanoiStroop_DrawAccuracy, \
                ExpH1HanoiDrawHanoiStroop_DrawSpeed, ExpH1HanoiDrawHanoiStroopInterruptionStats,
                ExpH1HanoiDrawHanoiStroopAccuracyStats, ExpH1HanoiDrawHanoiStroopSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH1HanoiDrawHanoiStroop_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ExpH1HanoiDrawHanoiStroopResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Experimental Group: H1: Hanoi-Path Tracing-Hanoi Stroop'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ExpH1HanoiDrawHanoiStroopResumptions, ExpH1HanoiDrawHanoiStroopResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH1HanoiDrawHanoiStroop_HanoiAccuracy,
                ExpH1HanoiDrawHanoiStroop_HanoiSpeed, ExpH1HanoiDrawHanoiStroop_DrawAccuracy, \
                ExpH1HanoiDrawHanoiStroop_DrawSpeed, ExpH1HanoiDrawHanoiStroopResumptionStats,
                ExpH1HanoiDrawHanoiStroopAccuracyStats, ExpH1HanoiDrawHanoiStroopSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ExpH1HanoiDrawHanoiStroopResumptionLag"
        # ExpH1HanoiDrawHanoiStroopResumptionStatsDF = pd.DataFrame(ExpH1HanoiDrawHanoiStroopResumptionStats)
        # ExpH1HanoiDrawHanoiStroopResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1HanoiDrawHanoiStroopResumptionStats.insert(0, "Name", "ExpH1HanoiDrawHanoiStroopResumptionLag")

        # filenameForStats = "ExpH1HanoiDrawHanoiStroopInterruptionLag"
        # ExpH1HanoiDrawHanoiStroopInterruptionStatsDF = pd.DataFrame(ExpH1HanoiDrawHanoiStroopInterruptionStats)
        # ExpH1HanoiDrawHanoiStroopInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1HanoiDrawHanoiStroopInterruptionStats.insert(0, "Name", "ExpH1HanoiDrawHanoiStroopInterruptionLag")

        # filenameForStats = "ExpH1HanoiDrawHanoiStroopAccuracy"
        # ExpH1HanoiDrawHanoiStroopAccuracyStatsDF = pd.DataFrame(ExpH1HanoiDrawHanoiStroopAccuracyStats)
        # ExpH1HanoiDrawHanoiStroopAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1HanoiDrawHanoiStroopAccuracyStats.insert(0, "Name", "ExpH1HanoiDrawHanoiStroopAccuracy")

        # filenameForStats = "ExpH1HanoiDrawHanoiStroopSpeed"
        # ExpH1HanoiDrawHanoiStroopSpeedStatsDF = pd.DataFrame(ExpH1HanoiDrawHanoiStroopSpeedStats)
        # ExpH1HanoiDrawHanoiStroopSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1HanoiDrawHanoiStroopSpeedStats.insert(0, "Name", "ExpH1HanoiDrawHanoiStroopSpeed")

    if p.group == 0 and p.hypotheses == 1 and p.starting_task == 2 and p.starting_interruption == 2:
        print("ExpH1HanoiDrawHanoiMath")
        # Calling sortStackAverageStatisticize method
        ExpH1HanoiDrawHanoiMathAttentions, \
        ExpH1HanoiDrawHanoiMathResumptions, \
        ExpH1HanoiDrawHanoiMathAccuracies,\
        ExpH1HanoiDrawHanoiMathSpeeds,\
        ExpH1HanoiDrawHanoiMathResumptionStats, \
        ExpH1HanoiDrawHanoiMathInterruptionStats, \
        ExpH1HanoiDrawHanoiMathAccuracyStats, \
        ExpH1HanoiDrawHanoiMathSpeedStats, \
        ExpH1HanoiDrawHanoiMathDiffResumption, \
        ExpH1HanoiDrawHanoiMathDiffInterruption, \
        ExpH1HanoiDrawHanoiMathDiffAccuracy, \
        ExpH1HanoiDrawHanoiMathDiffSpeed, \
        ExpH1HanoiDrawHanoiMathStackedAttentionList, \
        ExpH1HanoiDrawHanoiMathStackedResumptionList, \
        ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment, \
        ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting, \
        ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment, \
        ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting, \
        ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment, \
        ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting, \
        ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment, \
        ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ExpH1HanoiDrawHanoiMathSumResumptionLagsHanoiAssessment,
                                         ExpH1HanoiDrawHanoiMathSumResumptionLagsHanoiTesting,
                                         ExpH1HanoiDrawHanoiMathSumInterruptionLagsHanoiAssessment,
                                         ExpH1HanoiDrawHanoiMathSumInterruptionLagsHanoiTesting,
                                         ExpH1HanoiDrawHanoiMathstackedFlattenedAttentionList,
                                         ExpH1HanoiDrawHanoiMathstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ExpH1HanoiDrawHanoiMathcollectSumsMovesAndSequencesAssessment,
                                         ExpH1HanoiDrawHanoiMathcollectSumsMovesAndSequencesTesting,
                                         ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimesAssessment,
                                         ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ExpH1HanoiDrawHanoiMath_HanoiAccuracy, \
        ExpH1HanoiDrawHanoiMath_HanoiSpeed, \
        ExpH1HanoiDrawHanoiMath_DrawAccuracy, \
        ExpH1HanoiDrawHanoiMath_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ExpH1HanoiDrawHanoiMathcollectSumsMovesHanoiAssessment,
                ExpH1HanoiDrawHanoiMathcollectSumsMovesHanoiTraining,
                ExpH1HanoiDrawHanoiMathcollectSumsMovesHanoiTesting,
                ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiAssessment,
                ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiTraining,
                ExpH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiTesting,
                ExpH1HanoiDrawHanoiMathCollectCorrectnessDrawAssessment,
                ExpH1HanoiDrawHanoiMathCollectCorrectnessDrawTraining,
                ExpH1HanoiDrawHanoiMathCollectCorrectnessDrawTesting,
                ExpH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawAssessment,
                ExpH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawTraining,
                ExpH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawTesting)

        del ExpH1HanoiDrawHanoiMath_HanoiAccuracy[1]
        del ExpH1HanoiDrawHanoiMath_HanoiSpeed[1]
        del ExpH1HanoiDrawHanoiMath_DrawAccuracy[1]
        del ExpH1HanoiDrawHanoiMath_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH1HanoiDrawHanoiMath_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ExpH1HanoiDrawHanoiMathAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Experimental Group: H1: Hanoi-Path Tracing-Hanoi Math'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ExpH1HanoiDrawHanoiMathAttentions, ExpH1HanoiDrawHanoiMathAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH1HanoiDrawHanoiMath_HanoiAccuracy,
                ExpH1HanoiDrawHanoiMath_HanoiSpeed, ExpH1HanoiDrawHanoiMath_DrawAccuracy, \
                ExpH1HanoiDrawHanoiMath_DrawSpeed, ExpH1HanoiDrawHanoiMathInterruptionStats,
                ExpH1HanoiDrawHanoiMathAccuracyStats, ExpH1HanoiDrawHanoiMathSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH1HanoiDrawHanoiMath_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ExpH1HanoiDrawHanoiMathResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Experimental Group: H1: Hanoi-Path Tracing-Hanoi Math'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ExpH1HanoiDrawHanoiMathResumptions, ExpH1HanoiDrawHanoiMathResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH1HanoiDrawHanoiMath_HanoiAccuracy,
                ExpH1HanoiDrawHanoiMath_HanoiSpeed, ExpH1HanoiDrawHanoiMath_DrawAccuracy, \
                ExpH1HanoiDrawHanoiMath_DrawSpeed, ExpH1HanoiDrawHanoiMathResumptionStats,
                ExpH1HanoiDrawHanoiMathAccuracyStats, ExpH1HanoiDrawHanoiMathSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ExpH1HanoiDrawHanoiMathResumptionLag"
        # ExpH1HanoiDrawHanoiMathResumptionStatsDF = pd.DataFrame(ExpH1HanoiDrawHanoiMathResumptionStats)
        # ExpH1HanoiDrawHanoiMathResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1HanoiDrawHanoiMathResumptionStats.insert(0, "Name", "ExpH1HanoiDrawHanoiMathResumptionLag")

        # filenameForStats = "ExpH1HanoiDrawHanoiMathInterruptionLag"
        # ExpH1HanoiDrawHanoiMathInterruptionStatsDF = pd.DataFrame(ExpH1HanoiDrawHanoiMathInterruptionStats)
        # ExpH1HanoiDrawHanoiMathInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1HanoiDrawHanoiMathInterruptionStats.insert(0, "Name", "ExpH1HanoiDrawHanoiMathInterruptionLag")

        # filenameForStats = "ExpH1HanoiDrawHanoiMathAccuracy"
        # ExpH1HanoiDrawHanoiMathAccuracyStatsDF = pd.DataFrame(ExpH1HanoiDrawHanoiMathAccuracyStats)
        # ExpH1HanoiDrawHanoiMathAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1HanoiDrawHanoiMathAccuracyStats.insert(0, "Name", "ExpH1HanoiDrawHanoiMathAccuracy")

        # filenameForStats = "ExpH1HanoiDrawHanoiMathSpeed"
        # ExpH1HanoiDrawHanoiMathSpeedStatsDF = pd.DataFrame(ExpH1HanoiDrawHanoiMathSpeedStats)
        # ExpH1HanoiDrawHanoiMathSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH1HanoiDrawHanoiMathSpeedStats.insert(0, "Name", "ExpH1HanoiDrawHanoiMathSpeed")

    if p.group == 0 and p.hypotheses == 2 and p.starting_task == 1 and p.starting_interruption == 1:
        print("ExpH2StroopMathStroopDraw")
        # Calling sortStackAverageStatisticize method
        ExpH2StroopMathStroopDrawAttentions, \
        ExpH2StroopMathStroopDrawResumptions, \
        ExpH2StroopMathStroopDrawAccuracies, \
        ExpH2StroopMathStroopDrawSpeeds, \
        ExpH2StroopMathStroopDrawResumptionStats, \
        ExpH2StroopMathStroopDrawInterruptionStats, \
        ExpH2StroopMathStroopDrawAccuracyStats, \
        ExpH2StroopMathStroopDrawSpeedStats, \
        ExpH2StroopMathStroopDrawDiffResumption, \
        ExpH2StroopMathStroopDrawDiffInterruption, \
        ExpH2StroopMathStroopDrawDiffAccuracy, \
        ExpH2StroopMathStroopDrawDiffSpeed, \
        ExpH2StroopMathStroopDrawStackedAttentionList, \
        ExpH2StroopMathStroopDrawStackedResumptionList, \
        ExpH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment, \
        ExpH2StroopMathStroopDrawCollectedSumResumptionLagsTesting, \
        ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment, \
        ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting, \
        ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment, \
        ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting, \
        ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment, \
        ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ExpH2StroopMathStroopDrawSumResumptionLagsHanoiAssessment,
                                         ExpH2StroopMathStroopDrawSumResumptionLagsHanoiTesting,
                                         ExpH2StroopMathStroopDrawSumInterruptionLagsHanoiAssessment,
                                         ExpH2StroopMathStroopDrawSumInterruptionLagsHanoiTesting,
                                         ExpH2StroopMathStroopDrawstackedFlattenedAttentionList,
                                         ExpH2StroopMathStroopDrawstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ExpH2StroopMathStroopDrawcollectSumsMovesAndSequencesAssessment,
                                         ExpH2StroopMathStroopDrawcollectSumsMovesAndSequencesTesting,
                                         ExpH2StroopMathStroopDrawcollectSumsCompletionTimesAssessment,
                                         ExpH2StroopMathStroopDrawcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ExpH2StroopMathStroopDraw_HanoiAccuracy, \
        ExpH2StroopMathStroopDraw_HanoiSpeed, \
        ExpH2StroopMathStroopDraw_DrawAccuracy, \
        ExpH2StroopMathStroopDraw_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ExpH2StroopMathStroopDrawcollectSumsMovesHanoiAssessment,
                ExpH2StroopMathStroopDrawcollectSumsMovesHanoiTraining,
                ExpH2StroopMathStroopDrawcollectSumsMovesHanoiTesting,
                ExpH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiAssessment,
                ExpH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiTraining,
                ExpH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiTesting,
                ExpH2StroopMathStroopDrawCollectCorrectnessDrawAssessment,
                ExpH2StroopMathStroopDrawCollectCorrectnessDrawTraining,
                ExpH2StroopMathStroopDrawCollectCorrectnessDrawTesting,
                ExpH2StroopMathStroopDrawCollectSumsCompletionTimeDrawAssessment,
                ExpH2StroopMathStroopDrawCollectSumsCompletionTimeDrawTraining,
                ExpH2StroopMathStroopDrawCollectSumsCompletionTimeDrawTesting)

        del ExpH2StroopMathStroopDraw_HanoiAccuracy[1]
        del ExpH2StroopMathStroopDraw_HanoiSpeed[1]
        del ExpH2StroopMathStroopDraw_DrawAccuracy[1]
        del ExpH2StroopMathStroopDraw_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH2StroopMathStroopDraw_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ExpH2StroopMathStroopDrawAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Experimental Group: H2: Stroop-Math-Stroop Path Tracing'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ExpH2StroopMathStroopDrawAttentions, ExpH2StroopMathStroopDrawAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH2StroopMathStroopDraw_HanoiAccuracy,
                ExpH2StroopMathStroopDraw_HanoiSpeed, ExpH2StroopMathStroopDraw_DrawAccuracy, \
                ExpH2StroopMathStroopDraw_DrawSpeed, ExpH2StroopMathStroopDrawInterruptionStats,
                ExpH2StroopMathStroopDrawAccuracyStats, ExpH2StroopMathStroopDrawSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH2StroopMathStroopDraw_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ExpH2StroopMathStroopDrawResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Experimental Group: H2: Stroop-Math-Stroop Path Tracing'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ExpH2StroopMathStroopDrawResumptions, ExpH2StroopMathStroopDrawResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH2StroopMathStroopDraw_HanoiAccuracy,
                ExpH2StroopMathStroopDraw_HanoiSpeed, ExpH2StroopMathStroopDraw_DrawAccuracy, \
                ExpH2StroopMathStroopDraw_DrawSpeed, ExpH2StroopMathStroopDrawResumptionStats,
                ExpH2StroopMathStroopDrawAccuracyStats, ExpH2StroopMathStroopDrawSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ExpH2StroopMathStroopDrawResumptionLag"
        # ExpH2StroopMathStroopDrawResumptionStatsDF = pd.DataFrame(ExpH2StroopMathStroopDrawResumptionStats)
        # ExpH2StroopMathStroopDrawResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2StroopMathStroopDrawResumptionStats.insert(0, "Name", "ExpH2StroopMathStroopDrawResumptionLag")

        # filenameForStats = "ExpH2StroopMathStroopDrawInterruptionLag"
        # ExpH2StroopMathStroopDrawInterruptionStatsDF = pd.DataFrame(ExpH2StroopMathStroopDrawInterruptionStats)
        # ExpH2StroopMathStroopDrawInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2StroopMathStroopDrawInterruptionStats.insert(0, "Name", "ExpH2StroopMathStroopDrawInterruptionLag")

        # filenameForStats = "ExpH2StroopMathStroopDrawAccuracy"
        # ExpH2StroopMathStroopDrawAccuracyStatsDF = pd.DataFrame(ExpH2StroopMathStroopDrawAccuracyStats)
        # ExpH2StroopMathStroopDrawAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2StroopMathStroopDrawAccuracyStats.insert(0, "Name", "ExpH2StroopMathStroopDrawAccuracy")

        # filenameForStats = "ExpH2StroopMathStroopDrawSpeed"
        # ExpH2StroopMathStroopDrawSpeedStatsDF = pd.DataFrame(ExpH2StroopMathStroopDrawSpeedStats)
        # ExpH2StroopMathStroopDrawSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2StroopMathStroopDrawSpeedStats.insert(0, "Name", "ExpH2StroopMathStroopDrawSpeed")

    if p.group == 0 and p.hypotheses == 2 and p.starting_task == 1 and p.starting_interruption == 2:
        print("ExpH2MathStroopMathDraw")
        # Calling sortStackAverageStatisticize method
        ExpH2MathStroopMathDrawAttentions, \
        ExpH2MathStroopMathDrawResumptions, \
        ExpH2MathStroopMathDrawAccuracies, \
        ExpH2MathStroopMathDrawSpeeds, \
        ExpH2MathStroopMathDrawResumptionStats, \
        ExpH2MathStroopMathDrawInterruptionStats, \
        ExpH2MathStroopMathDrawAccuracyStats, \
        ExpH2MathStroopMathDrawSpeedStats, \
        ExpH2MathStroopMathDrawDiffResumption, \
        ExpH2MathStroopMathDrawDiffInterruption, \
        ExpH2MathStroopMathDrawDiffAccuracy, \
        ExpH2MathStroopMathDrawDiffSpeed, \
        ExpH2MathStroopMathDrawStackedAttentionList, \
        ExpH2MathStroopMathDrawStackedResumptionList, \
        ExpH2MathStroopMathDrawCollectedSumResumptionLagsAssessment, \
        ExpH2MathStroopMathDrawCollectedSumResumptionLagsTesting, \
        ExpH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment, \
        ExpH2MathStroopMathDrawCollectedSumInterruptionLagsTesting, \
        ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment, \
        ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting, \
        ExpH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment, \
        ExpH2MathStroopMathDrawCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ExpH2MathStroopMathDrawSumResumptionLagsHanoiAssessment,
                                         ExpH2MathStroopMathDrawSumResumptionLagsHanoiTesting,
                                         ExpH2MathStroopMathDrawSumInterruptionLagsHanoiAssessment,
                                         ExpH2MathStroopMathDrawSumInterruptionLagsHanoiTesting,
                                         ExpH2MathStroopMathDrawstackedFlattenedAttentionList,
                                         ExpH2MathStroopMathDrawstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ExpH2MathStroopMathDrawcollectSumsMovesAndSequencesAssessment,
                                         ExpH2MathStroopMathDrawcollectSumsMovesAndSequencesTesting,
                                         ExpH2MathStroopMathDrawcollectSumsCompletionTimesAssessment,
                                         ExpH2MathStroopMathDrawcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ExpH2MathStroopMathDraw_HanoiAccuracy, \
        ExpH2MathStroopMathDraw_HanoiSpeed, \
        ExpH2MathStroopMathDraw_DrawAccuracy, \
        ExpH2MathStroopMathDraw_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ExpH2MathStroopMathDrawcollectSumsMovesHanoiAssessment,
                ExpH2MathStroopMathDrawcollectSumsMovesHanoiTraining,
                ExpH2MathStroopMathDrawcollectSumsMovesHanoiTesting,
                ExpH2MathStroopMathDrawcollectSumsCompletionTimeHanoiAssessment,
                ExpH2MathStroopMathDrawcollectSumsCompletionTimeHanoiTraining,
                ExpH2MathStroopMathDrawcollectSumsCompletionTimeHanoiTesting,
                ExpH2MathStroopMathDrawCollectCorrectnessDrawAssessment,
                ExpH2MathStroopMathDrawCollectCorrectnessDrawTraining,
                ExpH2MathStroopMathDrawCollectCorrectnessDrawTesting,
                ExpH2MathStroopMathDrawCollectSumsCompletionTimeDrawAssessment,
                ExpH2MathStroopMathDrawCollectSumsCompletionTimeDrawTraining,
                ExpH2MathStroopMathDrawCollectSumsCompletionTimeDrawTesting)

        del ExpH2MathStroopMathDraw_HanoiAccuracy[1]
        del ExpH2MathStroopMathDraw_HanoiSpeed[1]
        del ExpH2MathStroopMathDraw_DrawAccuracy[1]
        del ExpH2MathStroopMathDraw_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH2MathStroopMathDraw_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ExpH2MathStroopMathDrawAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Experimental Group: H2: Math-Stroop-Math Path Tracing'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ExpH2MathStroopMathDrawAttentions, ExpH2MathStroopMathDrawAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH2MathStroopMathDraw_HanoiAccuracy,
                ExpH2MathStroopMathDraw_HanoiSpeed, ExpH2MathStroopMathDraw_DrawAccuracy, \
                ExpH2MathStroopMathDraw_DrawSpeed, ExpH2MathStroopMathDrawInterruptionStats,
                ExpH2MathStroopMathDrawAccuracyStats, ExpH2MathStroopMathDrawSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH2MathStroopMathDraw_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ExpH2MathStroopMathDrawResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Experimental Group: H2: Math-Stroop-Math Path Tracing'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ExpH2MathStroopMathDrawResumptions, ExpH2MathStroopMathDrawResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH2MathStroopMathDraw_HanoiAccuracy,
                ExpH2MathStroopMathDraw_HanoiSpeed, ExpH2MathStroopMathDraw_DrawAccuracy, \
                ExpH2MathStroopMathDraw_DrawSpeed, ExpH2MathStroopMathDrawResumptionStats,
                ExpH2MathStroopMathDrawAccuracyStats, ExpH2MathStroopMathDrawSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ExpH2MathStroopMathDrawResumptionLag"
        # ExpH2MathStroopMathDrawResumptionStatsDF = pd.DataFrame(ExpH2MathStroopMathDrawResumptionStats)
        # ExpH2MathStroopMathDrawResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2MathStroopMathDrawResumptionStats.insert(0, "Name", "ExpH2MathStroopMathDrawResumptionLag")

        # filenameForStats = "ExpH2MathStroopMathDrawInterruptionLag"
        # ExpH2MathStroopMathDrawInterruptionStatsDF = pd.DataFrame(ExpH2MathStroopMathDrawInterruptionStats)
        # ExpH2MathStroopMathDrawInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2MathStroopMathDrawInterruptionStats.insert(0, "Name", "ExpH2MathStroopMathDrawInterruptionLag")

        # filenameForStats = "ExpH2MathStroopMathDrawAccuracy"
        # ExpH2MathStroopMathDrawAccuracyStatsDF = pd.DataFrame(ExpH2MathStroopMathDrawAccuracyStats)
        # ExpH2MathStroopMathDrawAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2MathStroopMathDrawAccuracyStats.insert(0, "Name", "ExpH2MathStroopMathDrawAccuracy")

        # filenameForStats = "ExpH2MathStroopMathDrawSpeed"
        # ExpH2MathStroopMathDrawSpeedStatsDF = pd.DataFrame(ExpH2MathStroopMathDrawSpeedStats)
        # ExpH2MathStroopMathDrawSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2MathStroopMathDrawSpeedStats.insert(0, "Name", "ExpH2MathStroopMathDrawSpeed")

    if p.group == 0 and p.hypotheses == 2 and p.starting_task == 2 and p.starting_interruption == 1:
        print("ExpH2StroopMathStroopHanoi")
        # Calling sortStackAverageStatisticize method
        ExpH2StroopMathStroopHanoiAttentions, \
        ExpH2StroopMathStroopHanoiResumptions, \
        ExpH2StroopMathStroopHanoiAccuracies, \
        ExpH2StroopMathStroopHanoiSpeeds, \
        ExpH2StroopMathStroopHanoiResumptionStats, \
        ExpH2StroopMathStroopHanoiInterruptionStats, \
        ExpH2StroopMathStroopHanoiAccuracyStats, \
        ExpH2StroopMathStroopHanoiSpeedStats, \
        ExpH2StroopMathStroopHanoiDiffResumption, \
        ExpH2StroopMathStroopHanoiDiffInterruption, \
        ExpH2StroopMathStroopHanoiDiffAccuracy, \
        ExpH2StroopMathStroopHanoiDiffSpeed, \
        ExpH2StroopMathStroopHanoiStackedAttentionList, \
        ExpH2StroopMathStroopHanoiStackedResumptionList, \
        ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment, \
        ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting, \
        ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment, \
        ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting, \
        ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment, \
        ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting, \
        ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment, \
        ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ExpH2StroopMathStroopHanoiSumResumptionLagsHanoiAssessment,
                                         ExpH2StroopMathStroopHanoiSumResumptionLagsHanoiTesting,
                                         ExpH2StroopMathStroopHanoiSumInterruptionLagsHanoiAssessment,
                                         ExpH2StroopMathStroopHanoiSumInterruptionLagsHanoiTesting,
                                         ExpH2StroopMathStroopHanoistackedFlattenedAttentionList,
                                         ExpH2StroopMathStroopHanoistackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ExpH2StroopMathStroopHanoicollectSumsMovesAndSequencesAssessment,
                                         ExpH2StroopMathStroopHanoicollectSumsMovesAndSequencesTesting,
                                         ExpH2StroopMathStroopHanoicollectSumsCompletionTimesAssessment,
                                         ExpH2StroopMathStroopHanoicollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ExpH2StroopMathStroopHanoi_HanoiAccuracy, \
        ExpH2StroopMathStroopHanoi_HanoiSpeed, \
        ExpH2StroopMathStroopHanoi_DrawAccuracy, \
        ExpH2StroopMathStroopHanoi_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ExpH2StroopMathStroopHanoicollectSumsMovesHanoiAssessment,
                ExpH2StroopMathStroopHanoicollectSumsMovesHanoiTraining,
                ExpH2StroopMathStroopHanoicollectSumsMovesHanoiTesting,
                ExpH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiAssessment,
                ExpH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiTraining,
                ExpH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiTesting,
                ExpH2StroopMathStroopHanoiCollectCorrectnessDrawAssessment,
                ExpH2StroopMathStroopHanoiCollectCorrectnessDrawTraining,
                ExpH2StroopMathStroopHanoiCollectCorrectnessDrawTesting,
                ExpH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawAssessment,
                ExpH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawTraining,
                ExpH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawTesting)

        del ExpH2StroopMathStroopHanoi_HanoiAccuracy[1]
        del ExpH2StroopMathStroopHanoi_HanoiSpeed[1]
        del ExpH2StroopMathStroopHanoi_DrawAccuracy[1]
        del ExpH2StroopMathStroopHanoi_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH2StroopMathStroopHanoi_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ExpH2StroopMathStroopHanoiAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Experimental Group: H2: Stroop-Math-Stroop Hanoi'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ExpH2StroopMathStroopHanoiAttentions, ExpH2StroopMathStroopHanoiAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH2StroopMathStroopHanoi_HanoiAccuracy,
                ExpH2StroopMathStroopHanoi_HanoiSpeed, ExpH2StroopMathStroopHanoi_DrawAccuracy, \
                ExpH2StroopMathStroopHanoi_DrawSpeed, ExpH2StroopMathStroopHanoiInterruptionStats,
                ExpH2StroopMathStroopHanoiAccuracyStats, ExpH2StroopMathStroopHanoiSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH2StroopMathStroopHanoi_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ExpH2StroopMathStroopHanoiResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Experimental Group: H2: Stroop-Math-Stroop Hanoi'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ExpH2StroopMathStroopHanoiResumptions, ExpH2StroopMathStroopHanoiResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH2StroopMathStroopHanoi_HanoiAccuracy,
                ExpH2StroopMathStroopHanoi_HanoiSpeed, ExpH2StroopMathStroopHanoi_DrawAccuracy, \
                ExpH2StroopMathStroopHanoi_DrawSpeed, ExpH2StroopMathStroopHanoiResumptionStats,
                ExpH2StroopMathStroopHanoiAccuracyStats, ExpH2StroopMathStroopHanoiSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ExpH2StroopMathStroopHanoiResumptionLag"
        # ExpH2StroopMathStroopHanoiResumptionStatsDF = pd.DataFrame(ExpH2StroopMathStroopHanoiResumptionStats)
        # ExpH2StroopMathStroopHanoiResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2StroopMathStroopHanoiResumptionStats.insert(0, "Name", "ExpH2StroopMathStroopHanoiResumptionLag")

        # filenameForStats = "ExpH2StroopMathStroopHanoiInterruptionLag"
        # ExpH2StroopMathStroopHanoiInterruptionStatsDF = pd.DataFrame(ExpH2StroopMathStroopHanoiInterruptionStats)
        # ExpH2StroopMathStroopHanoiInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2StroopMathStroopHanoiInterruptionStats.insert(0, "Name", "ExpH2StroopMathStroopHanoiInterruptionLag")

        # filenameForStats = "ExpH2StroopMathStroopHanoiAccuracy"
        # ExpH2StroopMathStroopHanoiAccuracyStatsDF = pd.DataFrame(ExpH2StroopMathStroopHanoiAccuracyStats)
        # ExpH2StroopMathStroopHanoiAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2StroopMathStroopHanoiAccuracyStats.insert(0, "Name", "ExpH2StroopMathStroopHanoiAccuracy")

        # filenameForStats = "ExpH2StroopMathStroopHanoiSpeed"
        # ExpH2StroopMathStroopHanoiSpeedStatsDF = pd.DataFrame(ExpH2StroopMathStroopHanoiSpeedStats)
        # ExpH2StroopMathStroopHanoiSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2StroopMathStroopHanoiSpeedStats.insert(0, "Name", "ExpH2StroopMathStroopHanoiSpeed")

    if p.group == 0 and p.hypotheses == 2 and p.starting_task == 2 and p.starting_interruption == 2:
        print("ExpH2MathStroopMathHanoi")
        # Calling sortStackAverageStatisticize method
        ExpH2MathStroopMathHanoiAttentions, \
        ExpH2MathStroopMathHanoiResumptions, \
        ExpH2MathStroopMathHanoiAccuracies, \
        ExpH2MathStroopMathHanoiSpeeds, \
        ExpH2MathStroopMathHanoiResumptionStats, \
        ExpH2MathStroopMathHanoiInterruptionStats, \
        ExpH2MathStroopMathHanoiAccuracyStats, \
        ExpH2MathStroopMathHanoiSpeedStats, \
        ExpH2MathStroopMathHanoiDiffResumption, \
        ExpH2MathStroopMathHanoiDiffInterruption, \
        ExpH2MathStroopMathHanoiDiffAccuracy, \
        ExpH2MathStroopMathHanoiDiffSpeed, \
        ExpH2MathStroopMathHanoiStackedAttentionList, \
        ExpH2MathStroopMathHanoiStackedResumptionList, \
        ExpH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment, \
        ExpH2MathStroopMathHanoiCollectedSumResumptionLagsTesting, \
        ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment, \
        ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting, \
        ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment, \
        ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting, \
        ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment, \
        ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ExpH2MathStroopMathHanoiSumResumptionLagsHanoiAssessment,
                                         ExpH2MathStroopMathHanoiSumResumptionLagsHanoiTesting,
                                         ExpH2MathStroopMathHanoiSumInterruptionLagsHanoiAssessment,
                                         ExpH2MathStroopMathHanoiSumInterruptionLagsHanoiTesting,
                                         ExpH2MathStroopMathHanoistackedFlattenedAttentionList,
                                         ExpH2MathStroopMathHanoistackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ExpH2MathStroopMathHanoicollectSumsMovesAndSequencesAssessment,
                                         ExpH2MathStroopMathHanoicollectSumsMovesAndSequencesTesting,
                                         ExpH2MathStroopMathHanoicollectSumsCompletionTimesAssessment,
                                         ExpH2MathStroopMathHanoicollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ExpH2MathStroopMathHanoi_HanoiAccuracy, \
        ExpH2MathStroopMathHanoi_HanoiSpeed, \
        ExpH2MathStroopMathHanoi_DrawAccuracy, \
        ExpH2MathStroopMathHanoi_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ExpH2MathStroopMathHanoicollectSumsMovesHanoiAssessment,
                ExpH2MathStroopMathHanoicollectSumsMovesHanoiTraining,
                ExpH2MathStroopMathHanoicollectSumsMovesHanoiTesting,
                ExpH2MathStroopMathHanoicollectSumsCompletionTimeHanoiAssessment,
                ExpH2MathStroopMathHanoicollectSumsCompletionTimeHanoiTraining,
                ExpH2MathStroopMathHanoicollectSumsCompletionTimeHanoiTesting,
                ExpH2MathStroopMathHanoiCollectCorrectnessDrawAssessment,
                ExpH2MathStroopMathHanoiCollectCorrectnessDrawTraining,
                ExpH2MathStroopMathHanoiCollectCorrectnessDrawTesting,
                ExpH2MathStroopMathHanoiCollectSumsCompletionTimeDrawAssessment,
                ExpH2MathStroopMathHanoiCollectSumsCompletionTimeDrawTraining,
                ExpH2MathStroopMathHanoiCollectSumsCompletionTimeDrawTesting)

        del ExpH2MathStroopMathHanoi_HanoiAccuracy[1]
        del ExpH2MathStroopMathHanoi_HanoiSpeed[1]
        del ExpH2MathStroopMathHanoi_DrawAccuracy[1]
        del ExpH2MathStroopMathHanoi_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH2MathStroopMathHanoi_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ExpH2MathStroopMathHanoiAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Experimental Group: H2: Math-Stroop-Math Hanoi'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ExpH2MathStroopMathHanoiAttentions, ExpH2MathStroopMathHanoiAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH2MathStroopMathHanoi_HanoiAccuracy,
                ExpH2MathStroopMathHanoi_HanoiSpeed, ExpH2MathStroopMathHanoi_DrawAccuracy, \
                ExpH2MathStroopMathHanoi_DrawSpeed, ExpH2MathStroopMathHanoiInterruptionStats,
                ExpH2MathStroopMathHanoiAccuracyStats, ExpH2MathStroopMathHanoiSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ExpH2MathStroopMathHanoi_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ExpH2MathStroopMathHanoiResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Experimental Group: H2: Math-Stroop-Math Hanoi'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ExpH2MathStroopMathHanoiResumptions, ExpH2MathStroopMathHanoiResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ExpH2MathStroopMathHanoi_HanoiAccuracy,
                ExpH2MathStroopMathHanoi_HanoiSpeed, ExpH2MathStroopMathHanoi_DrawAccuracy, \
                ExpH2MathStroopMathHanoi_DrawSpeed, ExpH2MathStroopMathHanoiResumptionStats,
                ExpH2MathStroopMathHanoiAccuracyStats, ExpH2MathStroopMathHanoiSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ExpH2MathStroopMathHanoiResumptionLag"
        # ExpH2MathStroopMathHanoiResumptionStatsDF = pd.DataFrame(ExpH2MathStroopMathHanoiResumptionStats)
        # ExpH2MathStroopMathHanoiResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2MathStroopMathHanoiResumptionStats.insert(0, "Name", "ExpH2MathStroopMathHanoiResumptionLag")

        # filenameForStats = "ExpH2MathStroopMathHanoiInterruptionLag"
        # ExpH2MathStroopMathHanoiInterruptionStatsDF = pd.DataFrame(ExpH2MathStroopMathHanoiInterruptionStats)
        # ExpH2MathStroopMathHanoiInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2MathStroopMathHanoiInterruptionStats.insert(0, "Name", "ExpH2MathStroopMathHanoiInterruptionLag")

        # filenameForStats = "ExpH2MathStroopMathHanoiAccuracy"
        # ExpH2MathStroopMathHanoiAccuracyStatsDF = pd.DataFrame(ExpH2MathStroopMathHanoiAccuracyStats)
        # ExpH2MathStroopMathHanoiAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2MathStroopMathHanoiAccuracyStats.insert(0, "Name", "ExpH2MathStroopMathHanoiAccuracy")

        # filenameForStats = "ExpH2MathStroopMathHanoiSpeed"
        # ExpH2MathStroopMathHanoiSpeedStatsDF = pd.DataFrame(ExpH2MathStroopMathHanoiSpeedStats)
        # ExpH2MathStroopMathHanoiSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ExpH2MathStroopMathHanoiSpeedStats.insert(0, "Name", "ExpH2MathStroopMathHanoiSpeed")

    # Control Group Demarcation ------------------------------------------------------------------------

    if p.group == 1 and p.hypotheses == 1 and p.starting_task == 1 and p.starting_interruption == 1:
        # Control group = 1, hypothesis 1 (Task-Toggling) = p.hypotheses = 1
        print("ControlH1DrawHanoiDrawStroop")
        # Calling sortStackAverageStatisticize method
        ControlH1DrawHanoiDrawStroopAttentions, \
        ControlH1DrawHanoiDrawStroopResumptions, \
        ControlH1DrawHanoiDrawStroopAccuracies, \
        ControlH1DrawHanoiDrawStroopSpeeds, \
        ControlH1DrawHanoiDrawStroopResumptionStats, \
        ControlH1DrawHanoiDrawStroopInterruptionStats, \
        ControlH1DrawHanoiDrawStroopAccuracyStats, \
        ControlH1DrawHanoiDrawStroopSpeedStats, \
        ControlH1DrawHanoiDrawStroopDiffResumption, \
        ControlH1DrawHanoiDrawStroopDiffInterruption, \
        ControlH1DrawHanoiDrawStroopDiffAccuracy, \
        ControlH1DrawHanoiDrawStroopDiffSpeed, \
        ControlH1DrawHanoiDrawStroopStackedAttentionList, \
        ControlH1DrawHanoiDrawStroopStackedResumptionList, \
        ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment, \
        ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting, \
        ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment, \
        ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting, \
        ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment, \
        ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting, \
        ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment, \
        ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ControlH1DrawHanoiDrawStroopSumResumptionLagsHanoiAssessment,
                                         ControlH1DrawHanoiDrawStroopSumResumptionLagsHanoiTesting,
                                         ControlH1DrawHanoiDrawStroopSumInterruptionLagsHanoiAssessment,
                                         ControlH1DrawHanoiDrawStroopSumInterruptionLagsHanoiTesting,
                                         ControlH1DrawHanoiDrawStroopstackedFlattenedAttentionList,
                                         ControlH1DrawHanoiDrawStroopstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ControlH1DrawHanoiDrawStroopcollectSumsMovesAndSequencesAssessment,
                                         ControlH1DrawHanoiDrawStroopcollectSumsMovesAndSequencesTesting,
                                         ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimesAssessment,
                                         ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ControlH1DrawHanoiDrawStroop_HanoiAccuracy, \
        ControlH1DrawHanoiDrawStroop_HanoiSpeed, \
        ControlH1DrawHanoiDrawStroop_DrawAccuracy, \
        ControlH1DrawHanoiDrawStroop_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ControlH1DrawHanoiDrawStroopcollectSumsMovesHanoiAssessment,
                ControlH1DrawHanoiDrawStroopcollectSumsMovesHanoiTraining,
                ControlH1DrawHanoiDrawStroopcollectSumsMovesHanoiTesting,
                ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiAssessment,
                ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiTraining,
                ControlH1DrawHanoiDrawStroopcollectSumsCompletionTimeHanoiTesting,
                ControlH1DrawHanoiDrawStroopCollectCorrectnessDrawAssessment,
                ControlH1DrawHanoiDrawStroopCollectCorrectnessDrawTraining,
                ControlH1DrawHanoiDrawStroopCollectCorrectnessDrawTesting,
                ControlH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawAssessment,
                ControlH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawTraining,
                ControlH1DrawHanoiDrawStroopCollectSumsCompletionTimeDrawTesting)

        # Deleting the element corresponding to the value from the training phase in the control condition for plotting...
        # Take out index[1] for averagesMovesPerPhase_Accuracy & averagesTimesPerPhase_Speed only in control conditions
        del ControlH1DrawHanoiDrawStroop_HanoiAccuracy[1]
        del ControlH1DrawHanoiDrawStroop_HanoiSpeed[1]
        del ControlH1DrawHanoiDrawStroop_DrawAccuracy[1]
        del ControlH1DrawHanoiDrawStroop_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH1DrawHanoiDrawStroop_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ControlH1DrawHanoiDrawStroopAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Control Group: H1: Path Tracing-Hanoi-Path Tracing Stroop'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ControlH1DrawHanoiDrawStroopAttentions, ControlH1DrawHanoiDrawStroopAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH1DrawHanoiDrawStroop_HanoiAccuracy,
                ControlH1DrawHanoiDrawStroop_HanoiSpeed, ControlH1DrawHanoiDrawStroop_DrawAccuracy, \
                ControlH1DrawHanoiDrawStroop_DrawSpeed, ControlH1DrawHanoiDrawStroopInterruptionStats,
                ControlH1DrawHanoiDrawStroopAccuracyStats, ControlH1DrawHanoiDrawStroopSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH1DrawHanoiDrawStroop_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ControlH1DrawHanoiDrawStroopResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Control Group: H1: Path Tracing-Hanoi-Path Tracing Stroop'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ControlH1DrawHanoiDrawStroopResumptions, ControlH1DrawHanoiDrawStroopResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH1DrawHanoiDrawStroop_HanoiAccuracy,
                ControlH1DrawHanoiDrawStroop_HanoiSpeed, ControlH1DrawHanoiDrawStroop_DrawAccuracy, \
                ControlH1DrawHanoiDrawStroop_DrawSpeed, ControlH1DrawHanoiDrawStroopResumptionStats,
                ControlH1DrawHanoiDrawStroopAccuracyStats, ControlH1DrawHanoiDrawStroopSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ControlH1DrawHanoiDrawStroopResumptionLag"
        # ControlH1DrawHanoiDrawStroopResumptionStatsDF = pd.DataFrame(ControlH1DrawHanoiDrawStroopResumptionStats)
        # ControlH1DrawHanoiDrawStroopResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1DrawHanoiDrawStroopResumptionStats.insert(0, "Name", "ControlH1DrawHanoiDrawStroopResumptionLag")

        # filenameForStats = "ControlH1DrawHanoiDrawStroopInterruptionLag"
        # ControlH1DrawHanoiDrawStroopInterruptionStatsDF = pd.DataFrame(ControlH1DrawHanoiDrawStroopInterruptionStats)
        # ControlH1DrawHanoiDrawStroopInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1DrawHanoiDrawStroopInterruptionStats.insert(0, "Name", "ControlH1DrawHanoiDrawStroopInterruptionLag")

        # filenameForStats = "ControlH1DrawHanoiDrawStroopAccuracy"
        # ControlH1DrawHanoiDrawStroopAccuracyStatsDF = pd.DataFrame(ControlH1DrawHanoiDrawStroopAccuracyStats)
        # ControlH1DrawHanoiDrawStroopAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1DrawHanoiDrawStroopAccuracyStats.insert(0, "Name", "ControlH1DrawHanoiDrawStroopAccuracy")

        # filenameForStats = "ControlH1DrawHanoiDrawStroopSpeed"
        # ControlH1DrawHanoiDrawStroopSpeedStatsDF = pd.DataFrame(ControlH1DrawHanoiDrawStroopSpeedStats)
        # ControlH1DrawHanoiDrawStroopSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1DrawHanoiDrawStroopSpeedStats.insert(0, "Name", "ControlH1DrawHanoiDrawStroopSpeed")

    if p.group == 1 and p.hypotheses == 1 and p.starting_task == 1 and p.starting_interruption == 2:
        print("ControlH1DrawHanoiDrawMath")
        # Calling sortStackAverageStatisticize method
        ControlH1DrawHanoiDrawMathAttentions, \
        ControlH1DrawHanoiDrawMathResumptions, \
        ControlH1DrawHanoiDrawMathAccuracies, \
        ControlH1DrawHanoiDrawMathSpeeds, \
        ControlH1DrawHanoiDrawMathResumptionStats, \
        ControlH1DrawHanoiDrawMathInterruptionStats, \
        ControlH1DrawHanoiDrawMathAccuracyStats, \
        ControlH1DrawHanoiDrawMathSpeedStats, \
        ControlH1DrawHanoiDrawMathDiffResumption, \
        ControlH1DrawHanoiDrawMathDiffInterruption, \
        ControlH1DrawHanoiDrawMathDiffAccuracy, \
        ControlH1DrawHanoiDrawMathDiffSpeed, \
        ControlH1DrawHanoiDrawMathStackedAttentionList, \
        ControlH1DrawHanoiDrawMathStackedResumptionList, \
        ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment, \
        ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting, \
        ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment, \
        ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting, \
        ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment, \
        ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting, \
        ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment, \
        ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ControlH1DrawHanoiDrawMathSumResumptionLagsHanoiAssessment,
                                         ControlH1DrawHanoiDrawMathSumResumptionLagsHanoiTesting,
                                         ControlH1DrawHanoiDrawMathSumInterruptionLagsHanoiAssessment,
                                         ControlH1DrawHanoiDrawMathSumInterruptionLagsHanoiTesting,
                                         ControlH1DrawHanoiDrawMathstackedFlattenedAttentionList,
                                         ControlH1DrawHanoiDrawMathstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ControlH1DrawHanoiDrawMathcollectSumsMovesAndSequencesAssessment,
                                         ControlH1DrawHanoiDrawMathcollectSumsMovesAndSequencesTesting,
                                         ControlH1DrawHanoiDrawMathcollectSumsCompletionTimesAssessment,
                                         ControlH1DrawHanoiDrawMathcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ControlH1DrawHanoiDrawMath_HanoiAccuracy, \
        ControlH1DrawHanoiDrawMath_HanoiSpeed, \
        ControlH1DrawHanoiDrawMath_DrawAccuracy, \
        ControlH1DrawHanoiDrawMath_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ControlH1DrawHanoiDrawMathcollectSumsMovesHanoiAssessment,
                ControlH1DrawHanoiDrawMathcollectSumsMovesHanoiTraining,
                ControlH1DrawHanoiDrawMathcollectSumsMovesHanoiTesting,
                ControlH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiAssessment,
                ControlH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiTraining,
                ControlH1DrawHanoiDrawMathcollectSumsCompletionTimeHanoiTesting,
                ControlH1DrawHanoiDrawMathCollectCorrectnessDrawAssessment,
                ControlH1DrawHanoiDrawMathCollectCorrectnessDrawTraining,
                ControlH1DrawHanoiDrawMathCollectCorrectnessDrawTesting,
                ControlH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawAssessment,
                ControlH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawTraining,
                ControlH1DrawHanoiDrawMathCollectSumsCompletionTimeDrawTesting)

        # Deleting the element corresponding to the value from the training phase in the control condition for plotting...
        # Take out index[1] for averagesMovesPerPhase_Accuracy & averagesTimesPerPhase_Speed only in control conditions
        del ControlH1DrawHanoiDrawMath_HanoiAccuracy[1]
        del ControlH1DrawHanoiDrawMath_HanoiSpeed[1]
        del ControlH1DrawHanoiDrawMath_DrawAccuracy[1]
        del ControlH1DrawHanoiDrawMath_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH1DrawHanoiDrawMath_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ControlH1DrawHanoiDrawMathAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Control Group: H1: Path Tracing-Hanoi-Path Tracing Math'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ControlH1DrawHanoiDrawMathAttentions, ControlH1DrawHanoiDrawMathAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH1DrawHanoiDrawMath_HanoiAccuracy,
                ControlH1DrawHanoiDrawMath_HanoiSpeed, ControlH1DrawHanoiDrawMath_DrawAccuracy, \
                ControlH1DrawHanoiDrawMath_DrawSpeed, ControlH1DrawHanoiDrawMathInterruptionStats,
                ControlH1DrawHanoiDrawMathAccuracyStats, ControlH1DrawHanoiDrawMathSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH1DrawHanoiDrawMath_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ControlH1DrawHanoiDrawMathResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Control Group: H1: Path Tracing-Hanoi-Path Tracing Math'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ControlH1DrawHanoiDrawMathResumptions, ControlH1DrawHanoiDrawMathResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH1DrawHanoiDrawMath_HanoiAccuracy,
                ControlH1DrawHanoiDrawMath_HanoiSpeed, ControlH1DrawHanoiDrawMath_DrawAccuracy, \
                ControlH1DrawHanoiDrawMath_DrawSpeed, ControlH1DrawHanoiDrawMathResumptionStats,
                ControlH1DrawHanoiDrawMathAccuracyStats, ControlH1DrawHanoiDrawMathSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ControlH1DrawHanoiDrawMathResumptionLag"
        # ControlH1DrawHanoiDrawMathResumptionStatsDF = pd.DataFrame(ControlH1DrawHanoiDrawMathResumptionStats)
        # ControlH1DrawHanoiDrawMathResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1DrawHanoiDrawMathResumptionStats.insert(0, "Name", "ControlH1DrawHanoiDrawMathResumptionLag")

        # filenameForStats = "ControlH1DrawHanoiDrawMathInterruptionLag"
        # ControlH1DrawHanoiDrawMathInterruptionStatsDF = pd.DataFrame(ControlH1DrawHanoiDrawMathInterruptionStats)
        # ControlH1DrawHanoiDrawMathInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1DrawHanoiDrawMathInterruptionStats.insert(0, "Name", "ControlH1DrawHanoiDrawMathInterruptionLag")

        # filenameForStats = "ControlH1DrawHanoiDrawMathAccuracy"
        # ControlH1DrawHanoiDrawMathAccuracyStatsDF = pd.DataFrame(ControlH1DrawHanoiDrawMathAccuracyStats)
        # ControlH1DrawHanoiDrawMathAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1DrawHanoiDrawMathAccuracyStats.insert(0, "Name", "ControlH1DrawHanoiDrawMathAccuracy")

        # filenameForStats = "ControlH1DrawHanoiDrawMathSpeed"
        # ControlH1DrawHanoiDrawMathSpeedStatsDF = pd.DataFrame(ControlH1DrawHanoiDrawMathSpeedStats)
        # ControlH1DrawHanoiDrawMathSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1DrawHanoiDrawMathSpeedStats.insert(0, "Name", "ControlH1DrawHanoiDrawMathSpeed")

    if p.group == 1 and p.hypotheses == 1 and p.starting_task == 2 and p.starting_interruption == 1:
        print("ControlH1HanoiDrawHanoiStroop")
        # Calling sortStackAverageStatisticize method
        ControlH1HanoiDrawHanoiStroopAttentions, \
        ControlH1HanoiDrawHanoiStroopResumptions, \
        ControlH1HanoiDrawHanoiStroopAccuracies, \
        ControlH1HanoiDrawHanoiStroopSpeeds, \
        ControlH1HanoiDrawHanoiStroopResumptionStats, \
        ControlH1HanoiDrawHanoiStroopInterruptionStats, \
        ControlH1HanoiDrawHanoiStroopAccuracyStats, \
        ControlH1HanoiDrawHanoiStroopSpeedStats, \
        ControlH1HanoiDrawHanoiStroopDiffResumption, \
        ControlH1HanoiDrawHanoiStroopDiffInterruption, \
        ControlH1HanoiDrawHanoiStroopDiffAccuracy, \
        ControlH1HanoiDrawHanoiStroopDiffSpeed, \
        ControlH1HanoiDrawHanoiStroopStackedAttentionList, \
        ControlH1HanoiDrawHanoiStroopStackedResumptionList, \
        ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment, \
        ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting, \
        ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment, \
        ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting, \
        ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment, \
        ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting, \
        ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment, \
        ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ControlH1HanoiDrawHanoiStroopSumResumptionLagsHanoiAssessment,
                                         ControlH1HanoiDrawHanoiStroopSumResumptionLagsHanoiTesting,
                                         ControlH1HanoiDrawHanoiStroopSumInterruptionLagsHanoiAssessment,
                                         ControlH1HanoiDrawHanoiStroopSumInterruptionLagsHanoiTesting,
                                         ControlH1HanoiDrawHanoiStroopstackedFlattenedAttentionList,
                                         ControlH1HanoiDrawHanoiStroopstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ControlH1HanoiDrawHanoiStroopcollectSumsMovesAndSequencesAssessment,
                                         ControlH1HanoiDrawHanoiStroopcollectSumsMovesAndSequencesTesting,
                                         ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimesAssessment,
                                         ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ControlH1HanoiDrawHanoiStroop_HanoiAccuracy, \
        ControlH1HanoiDrawHanoiStroop_HanoiSpeed, \
        ControlH1HanoiDrawHanoiStroop_DrawAccuracy, \
        ControlH1HanoiDrawHanoiStroop_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ControlH1HanoiDrawHanoiStroopcollectSumsMovesHanoiAssessment,
                ControlH1HanoiDrawHanoiStroopcollectSumsMovesHanoiTraining,
                ControlH1HanoiDrawHanoiStroopcollectSumsMovesHanoiTesting,
                ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiAssessment,
                ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiTraining,
                ControlH1HanoiDrawHanoiStroopcollectSumsCompletionTimeHanoiTesting,
                ControlH1HanoiDrawHanoiStroopCollectCorrectnessDrawAssessment,
                ControlH1HanoiDrawHanoiStroopCollectCorrectnessDrawTraining,
                ControlH1HanoiDrawHanoiStroopCollectCorrectnessDrawTesting,
                ControlH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawAssessment,
                ControlH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawTraining,
                ControlH1HanoiDrawHanoiStroopCollectSumsCompletionTimeDrawTesting)

        # Deleting the element corresponding to the value from the training phase in the control condition for plotting...
        # Take out index[1] for averagesMovesPerPhase_Accuracy & averagesTimesPerPhase_Speed only in control conditions
        del ControlH1HanoiDrawHanoiStroop_HanoiAccuracy[1]
        del ControlH1HanoiDrawHanoiStroop_HanoiSpeed[1]
        del ControlH1HanoiDrawHanoiStroop_DrawAccuracy[1]
        del ControlH1HanoiDrawHanoiStroop_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH1HanoiDrawHanoiStroop_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ControlH1HanoiDrawHanoiStroopAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Control Group: H1: Hanoi-Path Tracing-Hanoi Stroop'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ControlH1HanoiDrawHanoiStroopAttentions, ControlH1HanoiDrawHanoiStroopAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH1HanoiDrawHanoiStroop_HanoiAccuracy,
                ControlH1HanoiDrawHanoiStroop_HanoiSpeed, ControlH1HanoiDrawHanoiStroop_DrawAccuracy, \
                ControlH1HanoiDrawHanoiStroop_DrawSpeed, ControlH1HanoiDrawHanoiStroopInterruptionStats,
                ControlH1HanoiDrawHanoiStroopAccuracyStats, ControlH1HanoiDrawHanoiStroopSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH1HanoiDrawHanoiStroop_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ControlH1HanoiDrawHanoiStroopResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Control Group: H1: Hanoi-Path Tracing-Hanoi Stroop'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ControlH1HanoiDrawHanoiStroopResumptions, ControlH1HanoiDrawHanoiStroopResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH1HanoiDrawHanoiStroop_HanoiAccuracy,
                ControlH1HanoiDrawHanoiStroop_HanoiSpeed, ControlH1HanoiDrawHanoiStroop_DrawAccuracy, \
                ControlH1HanoiDrawHanoiStroop_DrawSpeed, ControlH1HanoiDrawHanoiStroopResumptionStats,
                ControlH1HanoiDrawHanoiStroopAccuracyStats, ControlH1HanoiDrawHanoiStroopSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ControlH1HanoiDrawHanoiStroopResumptionLag"
        # ControlH1HanoiDrawHanoiStroopResumptionStatsDF = pd.DataFrame(ControlH1HanoiDrawHanoiStroopResumptionStats)
        # ControlH1HanoiDrawHanoiStroopResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1HanoiDrawHanoiStroopResumptionStats.insert(0, "Name", "ControlH1HanoiDrawHanoiStroopResumptionLag")

        # filenameForStats = "ControlH1HanoiDrawHanoiStroopInterruptionLag"
        # ControlH1HanoiDrawHanoiStroopInterruptionStatsDF = pd.DataFrame(ControlH1HanoiDrawHanoiStroopInterruptionStats)
        # ControlH1HanoiDrawHanoiStroopInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1HanoiDrawHanoiStroopInterruptionStats.insert(0, "Name", "ControlH1HanoiDrawHanoiStroopInterruptionLag")

        # filenameForStats = "ControlH1HanoiDrawHanoiStroopAccuracy"
        # ControlH1HanoiDrawHanoiStroopAccuracyStatsDF = pd.DataFrame(ControlH1HanoiDrawHanoiStroopAccuracyStats)
        # ControlH1HanoiDrawHanoiStroopAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1HanoiDrawHanoiStroopAccuracyStats.insert(0, "Name", "ControlH1HanoiDrawHanoiStroopAccuracy")

        # filenameForStats = "ControlH1HanoiDrawHanoiStroopSpeed"
        # ControlH1HanoiDrawHanoiStroopSpeedStatsDF = pd.DataFrame(ControlH1HanoiDrawHanoiStroopSpeedStats)
        # ControlH1HanoiDrawHanoiStroopSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1HanoiDrawHanoiStroopSpeedStats.insert(0, "Name", "ControlH1HanoiDrawHanoiStroopSpeed")

    if p.group == 1 and p.hypotheses == 1 and p.starting_task == 2 and p.starting_interruption == 2:
        print("ControlH1HanoiDrawHanoiMath")
        # Calling sortStackAverageStatisticize method
        ControlH1HanoiDrawHanoiMathAttentions, \
        ControlH1HanoiDrawHanoiMathResumptions, \
        ControlH1HanoiDrawHanoiMathAccuracies, \
        ControlH1HanoiDrawHanoiMathSpeeds, \
        ControlH1HanoiDrawHanoiMathResumptionStats, \
        ControlH1HanoiDrawHanoiMathInterruptionStats, \
        ControlH1HanoiDrawHanoiMathAccuracyStats, \
        ControlH1HanoiDrawHanoiMathSpeedStats, \
        ControlH1HanoiDrawHanoiMathDiffResumption, \
        ControlH1HanoiDrawHanoiMathDiffInterruption, \
        ControlH1HanoiDrawHanoiMathDiffAccuracy, \
        ControlH1HanoiDrawHanoiMathDiffSpeed, \
        ControlH1HanoiDrawHanoiMathStackedAttentionList, \
        ControlH1HanoiDrawHanoiMathStackedResumptionList, \
        ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment, \
        ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting, \
        ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment, \
        ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting, \
        ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment, \
        ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting, \
        ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment, \
        ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ControlH1HanoiDrawHanoiMathSumResumptionLagsHanoiAssessment,
                                         ControlH1HanoiDrawHanoiMathSumResumptionLagsHanoiTesting,
                                         ControlH1HanoiDrawHanoiMathSumInterruptionLagsHanoiAssessment,
                                         ControlH1HanoiDrawHanoiMathSumInterruptionLagsHanoiTesting,
                                         ControlH1HanoiDrawHanoiMathstackedFlattenedAttentionList,
                                         ControlH1HanoiDrawHanoiMathstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ControlH1HanoiDrawHanoiMathcollectSumsMovesAndSequencesAssessment,
                                         ControlH1HanoiDrawHanoiMathcollectSumsMovesAndSequencesTesting,
                                         ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimesAssessment,
                                         ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ControlH1HanoiDrawHanoiMath_HanoiAccuracy, \
        ControlH1HanoiDrawHanoiMath_HanoiSpeed, \
        ControlH1HanoiDrawHanoiMath_DrawAccuracy, \
        ControlH1HanoiDrawHanoiMath_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ControlH1HanoiDrawHanoiMathcollectSumsMovesHanoiAssessment,
                ControlH1HanoiDrawHanoiMathcollectSumsMovesHanoiTraining,
                ControlH1HanoiDrawHanoiMathcollectSumsMovesHanoiTesting,
                ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiAssessment,
                ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiTraining,
                ControlH1HanoiDrawHanoiMathcollectSumsCompletionTimeHanoiTesting,
                ControlH1HanoiDrawHanoiMathCollectCorrectnessDrawAssessment,
                ControlH1HanoiDrawHanoiMathCollectCorrectnessDrawTraining,
                ControlH1HanoiDrawHanoiMathCollectCorrectnessDrawTesting,
                ControlH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawAssessment,
                ControlH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawTraining,
                ControlH1HanoiDrawHanoiMathCollectSumsCompletionTimeDrawTesting)

        # Deleting the element corresponding to the value from the training phase in the control condition for plotting...
        # Take out index[1] for averagesMovesPerPhase_Accuracy & averagesTimesPerPhase_Speed only in control conditions
        del ControlH1HanoiDrawHanoiMath_HanoiAccuracy[1]
        del ControlH1HanoiDrawHanoiMath_HanoiSpeed[1]
        del ControlH1HanoiDrawHanoiMath_DrawAccuracy[1]
        del ControlH1HanoiDrawHanoiMath_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH1HanoiDrawHanoiMath_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ControlH1HanoiDrawHanoiMathAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Control Group: H1: Hanoi-Path Tracing-Hanoi Math'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ControlH1HanoiDrawHanoiMathAttentions, ControlH1HanoiDrawHanoiMathAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH1HanoiDrawHanoiMath_HanoiAccuracy,
                ControlH1HanoiDrawHanoiMath_HanoiSpeed, ControlH1HanoiDrawHanoiMath_DrawAccuracy, \
                ControlH1HanoiDrawHanoiMath_DrawSpeed, ControlH1HanoiDrawHanoiMathInterruptionStats,
                ControlH1HanoiDrawHanoiMathAccuracyStats, ControlH1HanoiDrawHanoiMathSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH1HanoiDrawHanoiMath_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ControlH1HanoiDrawHanoiMathResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Control Group: H1: Hanoi-Path Tracing-Hanoi Math'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ControlH1HanoiDrawHanoiMathResumptions, ControlH1HanoiDrawHanoiMathResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH1HanoiDrawHanoiMath_HanoiAccuracy,
                ControlH1HanoiDrawHanoiMath_HanoiSpeed, ControlH1HanoiDrawHanoiMath_DrawAccuracy, \
                ControlH1HanoiDrawHanoiMath_DrawSpeed, ControlH1HanoiDrawHanoiMathResumptionStats,
                ControlH1HanoiDrawHanoiMathAccuracyStats, ControlH1HanoiDrawHanoiMathSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ControlH1HanoiDrawHanoiMathResumptionLag"
        # ControlH1HanoiDrawHanoiMathResumptionStatsDF = pd.DataFrame(ControlH1HanoiDrawHanoiMathResumptionStats)
        # ControlH1HanoiDrawHanoiMathResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1HanoiDrawHanoiMathResumptionStats.insert(0, "Name", "ControlH1HanoiDrawHanoiMathResumptionLag")

        # filenameForStats = "ControlH1HanoiDrawHanoiMathInterruptionLag"
        # ControlH1HanoiDrawHanoiMathInterruptionStatsDF = pd.DataFrame(ControlH1HanoiDrawHanoiMathInterruptionStats)
        # ControlH1HanoiDrawHanoiMathInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1HanoiDrawHanoiMathInterruptionStats.insert(0, "Name", "ControlH1HanoiDrawHanoiMathInterruptionLag")

        # filenameForStats = "ControlH1HanoiDrawHanoiMathAccuracy"
        # ControlH1HanoiDrawHanoiMathAccuracyStatsDF = pd.DataFrame(ControlH1HanoiDrawHanoiMathAccuracyStats)
        # ControlH1HanoiDrawHanoiMathAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1HanoiDrawHanoiMathAccuracyStats.insert(0, "Name", "ControlH1HanoiDrawHanoiMathAccuracy")

        # filenameForStats = "ControlH1HanoiDrawHanoiMathSpeed"
        # ControlH1HanoiDrawHanoiMathSpeedStatsDF = pd.DataFrame(ControlH1HanoiDrawHanoiMathSpeedStats)
        # ControlH1HanoiDrawHanoiMathSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH1HanoiDrawHanoiMathSpeedStats.insert(0, "Name", "ControlH1HanoiDrawHanoiMathSpeed")

    if p.group == 1 and p.hypotheses == 2 and p.starting_task == 1 and p.starting_interruption == 1:
        print("ControlH2StroopMathStroopDraw")
        # Calling sortStackAverageStatisticize method
        ControlH2StroopMathStroopDrawAttentions, \
        ControlH2StroopMathStroopDrawResumptions, \
        ControlH2StroopMathStroopDrawAccuracies, \
        ControlH2StroopMathStroopDrawSpeeds, \
        ControlH2StroopMathStroopDrawResumptionStats, \
        ControlH2StroopMathStroopDrawInterruptionStats, \
        ControlH2StroopMathStroopDrawAccuracyStats, \
        ControlH2StroopMathStroopDrawSpeedStats, \
        ControlH2StroopMathStroopDrawDiffResumption, \
        ControlH2StroopMathStroopDrawDiffInterruption, \
        ControlH2StroopMathStroopDrawDiffAccuracy, \
        ControlH2StroopMathStroopDrawDiffSpeed, \
        ControlH2StroopMathStroopStackedAttentionList, \
        ControlH2StroopMathStroopStackedResumptionList, \
        ControlH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment, \
        ControlH2StroopMathStroopDrawCollectedSumResumptionLagsTesting, \
        ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment, \
        ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting, \
        ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment, \
        ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting, \
        ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment, \
        ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ControlH2StroopMathStroopDrawSumResumptionLagsHanoiAssessment,
                                         ControlH2StroopMathStroopDrawSumResumptionLagsHanoiTesting,
                                         ControlH2StroopMathStroopDrawSumInterruptionLagsHanoiAssessment,
                                         ControlH2StroopMathStroopDrawSumInterruptionLagsHanoiTesting,
                                         ControlH2StroopMathStroopDrawstackedFlattenedAttentionList,
                                         ControlH2StroopMathStroopDrawstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ControlH2StroopMathStroopDrawcollectSumsMovesAndSequencesAssessment,
                                         ControlH2StroopMathStroopDrawcollectSumsMovesAndSequencesTesting,
                                         ControlH2StroopMathStroopDrawcollectSumsCompletionTimesAssessment,
                                         ControlH2StroopMathStroopDrawcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ControlH2StroopMathStroopDraw_HanoiAccuracy, \
        ControlH2StroopMathStroopDraw_HanoiSpeed, \
        ControlH2StroopMathStroopDraw_DrawAccuracy, \
        ControlH2StroopMathStroopDraw_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ControlH2StroopMathStroopDrawcollectSumsMovesHanoiAssessment,
                ControlH2StroopMathStroopDrawcollectSumsMovesHanoiTraining,
                ControlH2StroopMathStroopDrawcollectSumsMovesHanoiTesting,
                ControlH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiAssessment,
                ControlH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiTraining,
                ControlH2StroopMathStroopDrawcollectSumsCompletionTimeHanoiTesting,
                ControlH2StroopMathStroopDrawCollectCorrectnessDrawAssessment,
                ControlH2StroopMathStroopDrawCollectCorrectnessDrawTraining,
                ControlH2StroopMathStroopDrawCollectCorrectnessDrawTesting,
                ControlH2StroopMathStroopDrawCollectSumsCompletionTimeDrawAssessment,
                ControlH2StroopMathStroopDrawCollectSumsCompletionTimeDrawTraining,
                ControlH2StroopMathStroopDrawCollectSumsCompletionTimeDrawTesting)

        # Deleting the element corresponding to the value from the training phase in the control condition for plotting...
        # Take out index[1] for averagesMovesPerPhase_Accuracy & averagesTimesPerPhase_Speed only in control conditions
        del ControlH2StroopMathStroopDraw_HanoiAccuracy[1]
        del ControlH2StroopMathStroopDraw_HanoiSpeed[1]
        del ControlH2StroopMathStroopDraw_DrawAccuracy[1]
        del ControlH2StroopMathStroopDraw_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH2StroopMathStroopDraw_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ControlH2StroopMathStroopDrawAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Control Group: H2: Stroop-Math-Stroop Path Tracing'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ControlH2StroopMathStroopDrawAttentions, ControlH2StroopMathStroopDrawAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH2StroopMathStroopDraw_HanoiAccuracy,
                ControlH2StroopMathStroopDraw_HanoiSpeed, ControlH2StroopMathStroopDraw_DrawAccuracy, \
                ControlH2StroopMathStroopDraw_DrawSpeed, ControlH2StroopMathStroopDrawInterruptionStats,
                ControlH2StroopMathStroopDrawAccuracyStats, ControlH2StroopMathStroopDrawSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH2StroopMathStroopDraw_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ControlH2StroopMathStroopDrawResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Control Group: H2: Stroop-Math-Stroop Path Tracing'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ControlH2StroopMathStroopDrawResumptions, ControlH2StroopMathStroopDrawResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH2StroopMathStroopDraw_HanoiAccuracy,
                ControlH2StroopMathStroopDraw_HanoiSpeed, ControlH2StroopMathStroopDraw_DrawAccuracy, \
                ControlH2StroopMathStroopDraw_DrawSpeed, ControlH2StroopMathStroopDrawResumptionStats,
                ControlH2StroopMathStroopDrawAccuracyStats, ControlH2StroopMathStroopDrawSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ControlH2StroopMathStroopDrawResumptionLag"
        # ControlH2StroopMathStroopDrawResumptionStatsDF = pd.DataFrame(ControlH2StroopMathStroopDrawResumptionStats)
        # ControlH2StroopMathStroopDrawResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2StroopMathStroopDrawResumptionStats.insert(0, "Name", "ControlH2StroopMathStroopDrawResumptionLag")

        # filenameForStats = "ControlH2StroopMathStroopDrawInterruptionLag"
        # ControlH2StroopMathStroopDrawInterruptionStatsDF = pd.DataFrame(ControlH2StroopMathStroopDrawInterruptionStats)
        # ControlH2StroopMathStroopDrawInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2StroopMathStroopDrawInterruptionStats.insert(0, "Name", "ControlH2StroopMathStroopDrawInterruptionLag")

        # filenameForStats = "ControlH2StroopMathStroopDrawAccuracy"
        # ControlH2StroopMathStroopDrawAccuracyStatsDF = pd.DataFrame(ControlH2StroopMathStroopDrawAccuracyStats)
        # ControlH2StroopMathStroopDrawAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2StroopMathStroopDrawAccuracyStats.insert(0, "Name", "ControlH2StroopMathStroopDrawAccuracy")

        # filenameForStats = "ControlH2StroopMathStroopDrawSpeed"
        # ControlH2StroopMathStroopDrawSpeedStatsDF = pd.DataFrame(ControlH2StroopMathStroopDrawSpeedStats)
        # ControlH2StroopMathStroopDrawSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2StroopMathStroopDrawSpeedStats.insert(0, "Name", "ControlH2StroopMathStroopDrawSpeed")

    if p.group == 1 and p.hypotheses == 2 and p.starting_task == 1 and p.starting_interruption == 2:
        print("ControlH2MathStroopMathDraw")
        # Calling sortStackAverageStatisticize method
        ControlH2MathStroopMathDrawAttentions, \
        ControlH2MathStroopMathDrawResumptions, \
        ControlH2MathStroopMathDrawAccuracies, \
        ControlH2MathStroopMathDrawSpeeds, \
        ControlH2MathStroopMathDrawResumptionStats, \
        ControlH2MathStroopMathDrawInterruptionStats, \
        ControlH2MathStroopMathDrawAccuracyStats, \
        ControlH2MathStroopMathDrawSpeedStats, \
        ControlH2MathStroopMathDrawDiffResumption, \
        ControlH2MathStroopMathDrawDiffInterruption, \
        ControlH2MathStroopMathDrawDiffAccuracy, \
        ControlH2MathStroopMathDrawDiffSpeed, \
        ControlH2MathStroopMathDrawStackedAttentionList, \
        ControlH2MathStroopMathDrawStackedResumptionList, \
        ControlH2MathStroopMathDrawCollectedSumResumptionLagsAssessment, \
        ControlH2MathStroopMathDrawCollectedSumResumptionLagsTesting, \
        ControlH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment, \
        ControlH2MathStroopMathDrawCollectedSumInterruptionLagsTesting, \
        ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment, \
        ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting, \
        ControlH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment, \
        ControlH2MathStroopMathDrawCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ControlH2MathStroopMathDrawSumResumptionLagsHanoiAssessment,
                                         ControlH2MathStroopMathDrawSumResumptionLagsHanoiTesting,
                                         ControlH2MathStroopMathDrawSumInterruptionLagsHanoiAssessment,
                                         ControlH2MathStroopMathDrawSumInterruptionLagsHanoiTesting,
                                         ControlH2MathStroopMathDrawstackedFlattenedAttentionList,
                                         ControlH2MathStroopMathDrawstackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ControlH2MathStroopMathDrawcollectSumsMovesAndSequencesAssessment,
                                         ControlH2MathStroopMathDrawcollectSumsMovesAndSequencesTesting,
                                         ControlH2MathStroopMathDrawcollectSumsCompletionTimesAssessment,
                                         ControlH2MathStroopMathDrawcollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ControlH2MathStroopMathDraw_HanoiAccuracy, \
        ControlH2MathStroopMathDraw_HanoiSpeed, \
        ControlH2MathStroopMathDraw_DrawAccuracy, \
        ControlH2MathStroopMathDraw_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ControlH2MathStroopMathDrawcollectSumsMovesHanoiAssessment,
                ControlH2MathStroopMathDrawcollectSumsMovesHanoiTraining,
                ControlH2MathStroopMathDrawcollectSumsMovesHanoiTesting,
                ControlH2MathStroopMathDrawcollectSumsCompletionTimeHanoiAssessment,
                ControlH2MathStroopMathDrawcollectSumsCompletionTimeHanoiTraining,
                ControlH2MathStroopMathDrawcollectSumsCompletionTimeHanoiTesting,
                ControlH2MathStroopMathDrawCollectCorrectnessDrawAssessment,
                ControlH2MathStroopMathDrawCollectCorrectnessDrawTraining,
                ControlH2MathStroopMathDrawCollectCorrectnessDrawTesting,
                ControlH2MathStroopMathDrawCollectSumsCompletionTimeDrawAssessment,
                ControlH2MathStroopMathDrawCollectSumsCompletionTimeDrawTraining,
                ControlH2MathStroopMathDrawCollectSumsCompletionTimeDrawTesting)

        # Deleting the element corresponding to the value from the training phase in the control condition for plotting...
        # Take out index[1] for averagesMovesPerPhase_Accuracy & averagesTimesPerPhase_Speed only in control conditions
        del ControlH2MathStroopMathDraw_HanoiAccuracy[1]
        del ControlH2MathStroopMathDraw_HanoiSpeed[1]
        del ControlH2MathStroopMathDraw_DrawAccuracy[1]
        del ControlH2MathStroopMathDraw_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH2MathStroopMathDraw_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ControlH2MathStroopMathDrawAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Control Group: H2: Math-Stroop-Math Path Tracing'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ControlH2MathStroopMathDrawAttentions, ControlH2MathStroopMathDrawAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH2MathStroopMathDraw_HanoiAccuracy,
                ControlH2MathStroopMathDraw_HanoiSpeed, ControlH2MathStroopMathDraw_DrawAccuracy, \
                ControlH2MathStroopMathDraw_DrawSpeed, ControlH2MathStroopMathDrawInterruptionStats,
                ControlH2MathStroopMathDrawAccuracyStats, ControlH2MathStroopMathDrawSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH2MathStroopMathDraw_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ControlH2MathStroopMathDrawResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Control Group: H2: Math-Stroop-Math Path tracing'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ControlH2MathStroopMathDrawResumptions, ControlH2MathStroopMathDrawResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH2MathStroopMathDraw_HanoiAccuracy,
                ControlH2MathStroopMathDraw_HanoiSpeed, ControlH2MathStroopMathDraw_DrawAccuracy, \
                ControlH2MathStroopMathDraw_DrawSpeed, ControlH2MathStroopMathDrawResumptionStats,
                ControlH2MathStroopMathDrawAccuracyStats, ControlH2MathStroopMathDrawSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ControlH2MathStroopMathDrawResumptionLag"
        # ControlH2MathStroopMathDrawResumptionStatsDF = pd.DataFrame(ControlH2MathStroopMathDrawResumptionStats)
        # ControlH2MathStroopMathDrawResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2MathStroopMathDrawResumptionStats.insert(0, "Name", "ControlH2MathStroopMathDrawResumptionLag")

        # filenameForStats = "ControlH2MathStroopMathDrawInterruptionLag"
        # ControlH2MathStroopMathDrawInterruptionStatsDF = pd.DataFrame(ControlH2MathStroopMathDrawInterruptionStats)
        # ControlH2MathStroopMathDrawInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2MathStroopMathDrawInterruptionStats.insert(0, "Name", "ControlH2MathStroopMathDrawInterruptionLag")

        # filenameForStats = "ControlH2MathStroopMathDrawAccuracy"
        # ControlH2MathStroopMathDrawAccuracyStatsDF = pd.DataFrame(ControlH2MathStroopMathDrawAccuracyStats)
        # ControlH2MathStroopMathDrawAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2MathStroopMathDrawAccuracyStats.insert(0, "Name", "ControlH2MathStroopMathDrawAccuracy")

        # filenameForStats = "ControlH2MathStroopMathDrawSpeed"
        # ControlH2MathStroopMathDrawSpeedStatsDF = pd.DataFrame(ControlH2MathStroopMathDrawSpeedStats)
        # ControlH2MathStroopMathDrawSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2MathStroopMathDrawSpeedStats.insert(0, "Name", "ControlH2MathStroopMathDrawSpeed")

    if p.group == 1 and p.hypotheses == 2 and p.starting_task == 2 and p.starting_interruption == 1:
        print("ControlH2StroopMathStroopHanoi")
        # Calling sortStackAverageStatisticize method
        ControlH2StroopMathStroopHanoiAttentions, \
        ControlH2StroopMathStroopHanoiResumptions, \
        ControlH2StroopMathStroopHanoiAccuracies, \
        ControlH2StroopMathStroopHanoiSpeeds, \
        ControlH2StroopMathStroopHanoiResumptionStats, \
        ControlH2StroopMathStroopHanoiInterruptionStats, \
        ControlH2StroopMathStroopHanoiAccuracyStats, \
        ControlH2StroopMathStroopHanoiSpeedStats, \
        ControlH2StroopMathStroopHanoiDiffResumption, \
        ControlH2StroopMathStroopHanoiDiffInterruption, \
        ControlH2StroopMathStroopHanoiDiffAccuracy, \
        ControlH2StroopMathStroopHanoiDiffSpeed, \
        ControlH2StroopMathStroopHanoiStackedAttentionList, \
        ControlH2StroopMathStroopHanoiStackedResumptionList, \
        ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment, \
        ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting, \
        ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment, \
        ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting, \
        ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment, \
        ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting, \
        ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment, \
        ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ControlH2StroopMathStroopHanoiSumResumptionLagsHanoiAssessment,
                                         ControlH2StroopMathStroopHanoiSumResumptionLagsHanoiTesting,
                                         ControlH2StroopMathStroopHanoiSumInterruptionLagsHanoiAssessment,
                                         ControlH2StroopMathStroopHanoiSumInterruptionLagsHanoiTesting,
                                         ControlH2StroopMathStroopHanoistackedFlattenedAttentionList,
                                         ControlH2StroopMathStroopHanoistackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ControlH2StroopMathStroopHanoicollectSumsMovesAndSequencesAssessment,
                                         ControlH2StroopMathStroopHanoicollectSumsMovesAndSequencesTesting,
                                         ControlH2StroopMathStroopHanoicollectSumsCompletionTimesAssessment,
                                         ControlH2StroopMathStroopHanoicollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ControlH2StroopMathStroopHanoi_HanoiAccuracy, \
        ControlH2StroopMathStroopHanoi_HanoiSpeed, \
        ControlH2StroopMathStroopHanoi_DrawAccuracy, \
        ControlH2StroopMathStroopHanoi_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ControlH2StroopMathStroopHanoicollectSumsMovesHanoiAssessment,
                ControlH2StroopMathStroopHanoicollectSumsMovesHanoiTraining,
                ControlH2StroopMathStroopHanoicollectSumsMovesHanoiTesting,
                ControlH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiAssessment,
                ControlH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiTraining,
                ControlH2StroopMathStroopHanoicollectSumsCompletionTimeHanoiTesting,
                ControlH2StroopMathStroopHanoiCollectCorrectnessDrawAssessment,
                ControlH2StroopMathStroopHanoiCollectCorrectnessDrawTraining,
                ControlH2StroopMathStroopHanoiCollectCorrectnessDrawTesting,
                ControlH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawAssessment,
                ControlH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawTraining,
                ControlH2StroopMathStroopHanoiCollectSumsCompletionTimeDrawTesting)

        # Deleting the element corresponding to the value from the training phase in the control condition for plotting...
        # Take out index[1] for averagesMovesPerPhase_Accuracy & averagesTimesPerPhase_Speed only in control conditions
        del ControlH2StroopMathStroopHanoi_HanoiAccuracy[1]
        del ControlH2StroopMathStroopHanoi_HanoiSpeed[1]
        del ControlH2StroopMathStroopHanoi_DrawAccuracy[1]
        del ControlH2StroopMathStroopHanoi_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH2StroopMathStroopHanoi_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ControlH2StroopMathStroopHanoiAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Control Group: H2: Stroop-Math-Stroop Hanoi'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ControlH2StroopMathStroopHanoiAttentions, ControlH2StroopMathStroopHanoiAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH2StroopMathStroopHanoi_HanoiAccuracy,
                ControlH2StroopMathStroopHanoi_HanoiSpeed, ControlH2StroopMathStroopHanoi_DrawAccuracy, \
                ControlH2StroopMathStroopHanoi_DrawSpeed, ControlH2StroopMathStroopHanoiInterruptionStats,
                ControlH2StroopMathStroopHanoiAccuracyStats, ControlH2StroopMathStroopHanoiSpeedStats)

        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH2StroopMathStroopHanoi_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ControlH2StroopMathStroopHanoiResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Control Group: H2: Stroop-Math-Stroop Hanoi'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ControlH2StroopMathStroopHanoiResumptions, ControlH2StroopMathStroopHanoiResumptions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH2StroopMathStroopHanoi_HanoiAccuracy,
                ControlH2StroopMathStroopHanoi_HanoiSpeed, ControlH2StroopMathStroopHanoi_DrawAccuracy, \
                ControlH2StroopMathStroopHanoi_DrawSpeed, ControlH2StroopMathStroopHanoiResumptionStats,
                ControlH2StroopMathStroopHanoiAccuracyStats, ControlH2StroopMathStroopHanoiSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        # filenameForStats = "ControlH2StroopMathStroopHanoiResumptionLag"
        # ControlH2StroopMathStroopHanoiResumptionStatsDF = pd.DataFrame(ControlH2StroopMathStroopHanoiResumptionStats)
        # ControlH2StroopMathStroopHanoiResumptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2StroopMathStroopHanoiResumptionStats.insert(0, "Name", "ControlH2StroopMathStroopHanoiResumptionLag")

        # filenameForStats = "ControlH2StroopMathStroopHanoiInterruptionLag"
        # ControlH2StroopMathStroopHanoiInterruptionStatsDF = pd.DataFrame(ControlH2StroopMathStroopHanoiInterruptionStats)
        # ControlH2StroopMathStroopHanoiInterruptionStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2StroopMathStroopHanoiInterruptionStats.insert(0, "Name", "ControlH2StroopMathStroopHanoiInterruptionLag")

        # filenameForStats = "ControlH2StroopMathStroopHanoiAccuracy"
        # ControlH2StroopMathStroopHanoiAccuracyStatsDF = pd.DataFrame(ControlH2StroopMathStroopHanoiAccuracyStats)
        # ControlH2StroopMathStroopHanoiAccuracyStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2StroopMathStroopHanoiAccuracyStats.insert(0, "Name", "ControlH2StroopMathStroopHanoiAccuracy")

        # filenameForStats = "ControlH2StroopMathStroopHanoiSpeed"
        # ControlH2StroopMathStroopHanoiSpeedStatsDF = pd.DataFrame(ControlH2StroopMathStroopHanoiSpeedStats)
        # ControlH2StroopMathStroopHanoiSpeedStatsDF.to_csv(
        #     '../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')
        ControlH2StroopMathStroopHanoiSpeedStats.insert(0, "Name", "ControlH2StroopMathStroopHanoiSpeed")

    if p.group == 1 and p.hypotheses == 2 and p.starting_task == 2 and p.starting_interruption == 2:
        print("ControlH2MathStroopMathHanoi")
        # Calling sortStackAverageStatisticize method
        ControlH2MathStroopMathHanoiAttentions, \
        ControlH2MathStroopMathHanoiResumptions, \
        ControlH2MathStroopMathHanoiAccuracies, \
        ControlH2MathStroopMathHanoiSpeeds, \
        ControlH2MathStroopMathHanoiResumptionStats,\
        ControlH2MathStroopMathHanoiInterruptionStats,\
        ControlH2MathStroopMathHanoiAccuracyStats,\
        ControlH2MathStroopMathHanoiSpeedStats,\
        ControlH2MathStroopMathHanoiDiffResumption,\
        ControlH2MathStroopMathHanoiDiffInterruption,\
        ControlH2MathStroopMathHanoiDiffAccuracy,\
        ControlH2MathStroopMathHanoiDiffSpeed,\
        ControlH2MathStroopMathHanoiStackedAttentionList,\
        ControlH2MathStroopMathHanoiStackedResumptionList, \
        ControlH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment, \
        ControlH2MathStroopMathHanoiCollectedSumResumptionLagsTesting, \
        ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment, \
        ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting, \
        ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment, \
        ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting, \
        ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment, \
        ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting = \
            sortStackAverageStatisticize(assessInterruptLagsList,
                                         trainingInterruptLagsList,
                                         testingInterruptLagsList,
                                         assessResumptionLagsList,
                                         trainingResumptionLagsList,
                                         testingResumptionLagsList,
                                         ControlH2MathStroopMathHanoiSumResumptionLagsHanoiAssessment,
                                         ControlH2MathStroopMathHanoiSumResumptionLagsHanoiTesting,
                                         ControlH2MathStroopMathHanoiSumInterruptionLagsHanoiAssessment,
                                         ControlH2MathStroopMathHanoiSumInterruptionLagsHanoiTesting,
                                         ControlH2MathStroopMathHanoistackedFlattenedAttentionList,
                                         ControlH2MathStroopMathHanoistackedFlattenedResumptionList,
                                         accuracyInAssessmentSum,
                                         speedInAssessmentSum,
                                         accuracyInTestingSum,
                                         speedInTestingSum,
                                         accuracyInAssessmentList,
                                         accuracyInTestingList,
                                         speedInAssessmentList,
                                         speedInTestingList,
                                         ControlH2MathStroopMathHanoicollectSumsMovesAndSequencesAssessment,
                                         ControlH2MathStroopMathHanoicollectSumsMovesAndSequencesTesting,
                                         ControlH2MathStroopMathHanoicollectSumsCompletionTimesAssessment,
                                         ControlH2MathStroopMathHanoicollectSumsCompletionTimesTesting)

        # Calling doAverages4SpeedAccuracyStatisticize Method
        ControlH2MathStroopMathHanoi_HanoiAccuracy, \
        ControlH2MathStroopMathHanoi_HanoiSpeed, \
        ControlH2MathStroopMathHanoi_DrawAccuracy, \
        ControlH2MathStroopMathHanoi_DrawSpeed = \
            doAverages4SpeedAccuracyStatisticize(
                accuracyInAssessmentSum,
                accuracyInTrainingSum,
                accuracyInTestingSum,
                speedInAssessmentSum,
                speedInTrainingSum,
                speedInTestingSum,
                ControlH2MathStroopMathHanoicollectSumsMovesHanoiAssessment,
                ControlH2MathStroopMathHanoicollectSumsMovesHanoiTraining,
                ControlH2MathStroopMathHanoicollectSumsMovesHanoiTesting,
                ControlH2MathStroopMathHanoicollectSumsCompletionTimeHanoiAssessment,
                ControlH2MathStroopMathHanoicollectSumsCompletionTimeHanoiTraining,
                ControlH2MathStroopMathHanoicollectSumsCompletionTimeHanoiTesting,
                ControlH2MathStroopMathHanoiCollectCorrectnessDrawAssessment,
                ControlH2MathStroopMathHanoiCollectCorrectnessDrawTraining,
                ControlH2MathStroopMathHanoiCollectCorrectnessDrawTesting,
                ControlH2MathStroopMathHanoiCollectSumsCompletionTimeDrawAssessment,
                ControlH2MathStroopMathHanoiCollectSumsCompletionTimeDrawTraining,
                ControlH2MathStroopMathHanoiCollectSumsCompletionTimeDrawTesting)

        # Deleting the element corresponding to the value from the training phase in the control condition for plotting...
        # Take out index[1] for averagesMovesPerPhase_Accuracy & averagesTimesPerPhase_Speed only in control conditions
        del ControlH2MathStroopMathHanoi_HanoiAccuracy[1]
        del ControlH2MathStroopMathHanoi_HanoiSpeed[1]
        del ControlH2MathStroopMathHanoi_DrawAccuracy[1]
        del ControlH2MathStroopMathHanoi_DrawSpeed[1]

        # Demarcation for code for avg interruption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH2MathStroopMathHanoi_AVG_InterruptionLags"
        averageAttentionsDF = pd.DataFrame(ControlH2MathStroopMathHanoiAttentions)
        averageAttentionsDF.to_csv('../DataResults/InterruptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/InterruptionLags/'
        title = 'Control Group: H2: Math-Stroop-Math Hanoi'
        yLabel = 'Average Interruption Lag Times (Seconds)'
        plotter(ControlH2MathStroopMathHanoiAttentions, ControlH2MathStroopMathHanoiAttentions,
                title, yLabel, PlotSpot, filenameForCharts, ControlH2MathStroopMathHanoi_HanoiAccuracy,
                ControlH2MathStroopMathHanoi_HanoiSpeed, ControlH2MathStroopMathHanoi_DrawAccuracy, \
        ControlH2MathStroopMathHanoi_DrawSpeed, ControlH2MathStroopMathHanoiInterruptionStats,
                ControlH2MathStroopMathHanoiAccuracyStats, ControlH2MathStroopMathHanoiSpeedStats)


        # Demarcation for code for avg resumption lag for participants across all phases within variation of condition

        filenameForCharts = "ControlH2MathStroopMathHanoi_AVG_ResumptionLags"
        averageResumptionsDF = pd.DataFrame(ControlH2MathStroopMathHanoiResumptions)
        averageResumptionsDF.to_csv('../DataResults/ResumptionLags/' + filenameForCharts + '.csv')

        PlotSpot = '../DataResults/ResumptionLags/'
        title = 'Control Group: H2: Math-Stroop-Math Hanoi'
        yLabel = 'Average  Resumption Lag Times (Seconds)'
        plotter(ControlH2MathStroopMathHanoiResumptions, ControlH2MathStroopMathHanoiResumptions,
                title, yLabel, PlotSpot, filenameForCharts,ControlH2MathStroopMathHanoi_HanoiAccuracy,
                ControlH2MathStroopMathHanoi_HanoiSpeed, ControlH2MathStroopMathHanoi_DrawAccuracy, \
        ControlH2MathStroopMathHanoi_DrawSpeed, ControlH2MathStroopMathHanoiResumptionStats,
                ControlH2MathStroopMathHanoiAccuracyStats, ControlH2MathStroopMathHanoiSpeedStats)

        # Demarcation for code for saving statistics within variation of condition

        ControlH2MathStroopMathHanoiResumptionStats.insert(0, "Name", "ControlH2MathStroopMathHanoiResumptionLag")

        ControlH2MathStroopMathHanoiInterruptionStats.insert(0, "Name", "ControlH2MathStroopMathHanoiInterruptionLag")

        ControlH2MathStroopMathHanoiAccuracyStats.insert(0, "Name", "ControlH2MathStroopMathHanoiAccuracy")

        ControlH2MathStroopMathHanoiSpeedStats.insert(0, "Name", "ControlH2MathStroopMathHanoiSpeed")

# Plot summarizing charts by hypotheses and overlay stats

PlotPlace = '../DataResults/Stats/SummarizingStats/'

# Top left quadrant
stackedAttentionListExpH1.append(ExpH1DrawHanoiDrawStroopAttentions)
stackedAttentionListExpH1.append(ExpH1DrawHanoiDrawMathAttentions)
stackedAttentionListExpH1.append(ExpH1HanoiDrawHanoiStroopAttentions)
stackedAttentionListExpH1.append(ExpH1HanoiDrawHanoiMathAttentions)
averageAttentionsLagListExpH1 = [sum(allH1Attentions) / len(stackedAttentionListExpH1) for
                         allH1Attentions in zip(*stackedAttentionListExpH1)]
first8averageAttentionsLagListExpH1 = averageAttentionsLagListExpH1[:8]

stackedResumptionListExpH1.append(ExpH1DrawHanoiDrawStroopResumptions)
stackedResumptionListExpH1.append(ExpH1DrawHanoiDrawMathResumptions)
stackedResumptionListExpH1.append(ExpH1HanoiDrawHanoiStroopResumptions)
stackedResumptionListExpH1.append(ExpH1HanoiDrawHanoiMathResumptions)
averageResumptionLagListExpH1 = [sum(allH1Resumptions) / len(stackedResumptionListExpH1) for
                         allH1Resumptions in zip(*stackedResumptionListExpH1)]

# Bottom left quadrant
stackedAttentionListExpH2.append(ExpH2StroopMathStroopDrawAttentions)
stackedAttentionListExpH2.append(ExpH2MathStroopMathDrawAttentions)
stackedAttentionListExpH2.append(ExpH2StroopMathStroopHanoiAttentions)
stackedAttentionListExpH2.append(ExpH2MathStroopMathHanoiAttentions)
averageAttentionsLagListExpH2 = [sum(allH1Attentions) / len(stackedAttentionListExpH2) for
                         allH1Attentions in zip(*stackedAttentionListExpH2)]
first8averageAttentionsLagListExpH2 = averageAttentionsLagListExpH2[:8]

stackedResumptionListExpH2.append(ExpH2StroopMathStroopDrawResumptions)
stackedResumptionListExpH2.append(ExpH2MathStroopMathDrawResumptions)
stackedResumptionListExpH2.append(ExpH2StroopMathStroopHanoiResumptions)
stackedResumptionListExpH2.append(ExpH2MathStroopMathHanoiResumptions)
averageResumptionLagListExpH2 = [sum(allH1Resumptions) / len(stackedResumptionListExpH2) for
                         allH1Resumptions in zip(*stackedResumptionListExpH2)]

# Control condition has higher highs in the assessment phases, and lower lows in the testing phasers, resulting in
# larger differences between the assessment and testing phases when compared to the experimental condition

# Top right quadrant
stackedAttentionListControlH1.append(ControlH1DrawHanoiDrawStroopAttentions)
stackedAttentionListControlH1.append(ControlH1DrawHanoiDrawMathAttentions)
stackedAttentionListControlH1.append(ControlH1HanoiDrawHanoiStroopAttentions)
stackedAttentionListControlH1.append(ControlH1HanoiDrawHanoiMathAttentions)
averageAttentionsLagListControlH1 = [sum(allH1Attentions) / len(stackedAttentionListControlH1) for
                         allH1Attentions in zip(*stackedAttentionListControlH1)]
first8averageAttentionsLagListControlH1 = averageAttentionsLagListControlH1[:8]

first8averageAttentionsLagStatsH1 = statisticize.ttest(first8averageAttentionsLagListExpH1,
                                              first8averageAttentionsLagListControlH1, paired=False)#, alternative="greater")
filenameForStats = "first8averageAttentionsLagStatsTopQuadrants"
first8averageAttentionsLagStatsH1.to_csv('../DataResults/Stats/SummarizingStats/' + filenameForStats + '.csv')


stackedResumptionListControlH1.append(ControlH1DrawHanoiDrawStroopResumptions)
stackedResumptionListControlH1.append(ControlH1DrawHanoiDrawMathResumptions)
stackedResumptionListControlH1.append(ControlH1HanoiDrawHanoiStroopResumptions)
stackedResumptionListControlH1.append(ControlH1HanoiDrawHanoiMathResumptions)
averageResumptionLagListControlH1 = [sum(allH1Resumptions) / len(stackedResumptionListControlH1) for
                         allH1Resumptions in zip(*stackedResumptionListControlH1)]

# Bottom right quadrant
stackedAttentionListControlH2.append(ControlH2StroopMathStroopDrawAttentions)
stackedAttentionListControlH2.append(ControlH2MathStroopMathDrawAttentions)
stackedAttentionListControlH2.append(ControlH2StroopMathStroopHanoiAttentions)
stackedAttentionListControlH2.append(ControlH2MathStroopMathHanoiAttentions)
averageAttentionsLagListControlH2 = [sum(allH2Attentions) / len(stackedAttentionListControlH2) for
                         allH2Attentions in zip(*stackedAttentionListControlH2)]
first8averageAttentionsLagListControlH2 = averageAttentionsLagListControlH2[:8]

first8averageAttentionsLagStatsH2 = statisticize.ttest(first8averageAttentionsLagListExpH2,
                                              first8averageAttentionsLagListControlH2, paired=False)#, alternative="greater")
filenameForStats = "first8averageAttentionsLagStatsBottomQuadrants"
first8averageAttentionsLagStatsH2.to_csv('../DataResults/Stats/SummarizingStats/' + filenameForStats + '.csv')

stackedResumptionListControlH2.append(ControlH2StroopMathStroopDrawResumptions)
stackedResumptionListControlH2.append(ControlH2MathStroopMathDrawResumptions)
stackedResumptionListControlH2.append(ControlH2StroopMathStroopHanoiResumptions)
stackedResumptionListControlH2.append(ControlH2MathStroopMathHanoiResumptions)
averageResumptionLagListControlH2 = [sum(allH2Resumptions) / len(stackedResumptionListControlH2) for
                         allH2Resumptions in zip(*stackedResumptionListControlH2)]


IntraVariationalStatsEach = pd.concat([
    ExpH1DrawHanoiDrawStroopResumptionStats,
    ExpH1DrawHanoiDrawMathResumptionStats,
    ExpH1HanoiDrawHanoiStroopResumptionStats,
    ExpH1HanoiDrawHanoiMathResumptionStats,
    ExpH2StroopMathStroopDrawResumptionStats,
    ExpH2MathStroopMathDrawResumptionStats,
    ExpH2StroopMathStroopHanoiResumptionStats,
    ExpH2MathStroopMathHanoiResumptionStats,

    ExpH1DrawHanoiDrawStroopInterruptionStats,
    ExpH1DrawHanoiDrawMathInterruptionStats,
    ExpH1HanoiDrawHanoiStroopInterruptionStats,
    ExpH1HanoiDrawHanoiMathInterruptionStats,
    ExpH2StroopMathStroopDrawInterruptionStats,
    ExpH2MathStroopMathDrawInterruptionStats,
    ExpH2StroopMathStroopHanoiInterruptionStats,
    ExpH2MathStroopMathHanoiInterruptionStats,

    ExpH1DrawHanoiDrawStroopAccuracyStats,
    ExpH1DrawHanoiDrawMathAccuracyStats,
    ExpH1HanoiDrawHanoiStroopAccuracyStats,
    ExpH1HanoiDrawHanoiMathAccuracyStats,
    ExpH2StroopMathStroopDrawAccuracyStats,
    ExpH2MathStroopMathDrawAccuracyStats,
    ExpH2StroopMathStroopHanoiAccuracyStats,
    ExpH2MathStroopMathHanoiAccuracyStats,

    ExpH1DrawHanoiDrawStroopSpeedStats,
    ExpH1DrawHanoiDrawMathSpeedStats,
    ExpH1HanoiDrawHanoiStroopSpeedStats,
    ExpH1HanoiDrawHanoiMathSpeedStats,
    ExpH2StroopMathStroopDrawSpeedStats,
    ExpH2MathStroopMathDrawSpeedStats,
    ExpH2StroopMathStroopHanoiSpeedStats,
    ExpH2MathStroopMathHanoiSpeedStats,

    ControlH1DrawHanoiDrawStroopResumptionStats,
    ControlH1DrawHanoiDrawMathResumptionStats,
    ControlH1HanoiDrawHanoiStroopResumptionStats,
    ControlH1HanoiDrawHanoiMathResumptionStats,
    ControlH2StroopMathStroopDrawResumptionStats,
    ControlH2MathStroopMathDrawResumptionStats,
    ControlH2StroopMathStroopHanoiResumptionStats,
    ControlH2MathStroopMathHanoiResumptionStats,

    ControlH1DrawHanoiDrawStroopInterruptionStats,
    ControlH1DrawHanoiDrawMathInterruptionStats,
    ControlH1HanoiDrawHanoiStroopInterruptionStats,
    ControlH1HanoiDrawHanoiMathInterruptionStats,
    ControlH2StroopMathStroopDrawInterruptionStats,
    ControlH2MathStroopMathDrawInterruptionStats,
    ControlH2StroopMathStroopHanoiInterruptionStats,
    ControlH2MathStroopMathHanoiInterruptionStats,

    ControlH1DrawHanoiDrawStroopAccuracyStats,
    ControlH1DrawHanoiDrawMathAccuracyStats,
    ControlH1HanoiDrawHanoiStroopAccuracyStats,
    ControlH1HanoiDrawHanoiMathAccuracyStats,
    ControlH2StroopMathStroopDrawAccuracyStats,
    ControlH2MathStroopMathDrawAccuracyStats,
    ControlH2StroopMathStroopHanoiAccuracyStats,
    ControlH2MathStroopMathHanoiAccuracyStats,

    ControlH1DrawHanoiDrawStroopSpeedStats,
    ControlH1DrawHanoiDrawMathSpeedStats,
    ControlH1HanoiDrawHanoiStroopSpeedStats,
    ControlH1HanoiDrawHanoiMathSpeedStats,
    ControlH2StroopMathStroopDrawSpeedStats,
    ControlH2MathStroopMathDrawSpeedStats,
    ControlH2StroopMathStroopHanoiSpeedStats,
    ControlH2MathStroopMathHanoiSpeedStats
                                         ])
filenameForStats = "IntraVariationalStatsEach"
IntraVariationalStatsEach.to_csv('../DataResults/Stats/IntraVariationalStats/' + filenameForStats + '.csv')

# Independent Samples t-Test (automatically uses Welch T-test when the sample sizes are unequal)
# alternative='greater' to indicate that we test against the alternative hypothesis that the mean of Exp is greater than the mean of Control.

# ExpH1DrawHanoiDrawStroop to ControlH1DrawHanoiDrawStroop
H1DrawHanoiDrawStroopResumption = statisticize.ttest(ExpH1DrawHanoiDrawStroopDiffResumption,
                              ControlH1DrawHanoiDrawStroopDiffResumption, paired=False, alternative="greater")
H1DrawHanoiDrawStroopResumption.insert(0, "Name", "H1DrawHanoiDrawStroopResumption Btw Exp and Control")

H1DrawHanoiDrawStroopInterruption = statisticize.ttest(ExpH1DrawHanoiDrawStroopDiffInterruption,
                                              ControlH1DrawHanoiDrawStroopDiffInterruption, paired=False, alternative="greater")
H1DrawHanoiDrawStroopInterruption.insert(0, "Name", "H1DrawHanoiDrawStroopInterruption Btw Exp and Control")

H1DrawHanoiDrawStroopAccuracy = statisticize.ttest(ExpH1DrawHanoiDrawStroopDiffAccuracy,
                                              ControlH1DrawHanoiDrawStroopDiffAccuracy, paired=False, alternative="greater")
H1DrawHanoiDrawStroopAccuracy.insert(0, "Name", "H1DrawHanoiDrawStroopAccuracy Btw Exp and Control")

H1DrawHanoiDrawStroopSpeed = statisticize.ttest(ExpH1DrawHanoiDrawStroopDiffSpeed,
                                              ControlH1DrawHanoiDrawStroopDiffSpeed, paired=False, alternative="greater")
H1DrawHanoiDrawStroopSpeed.insert(0, "Name", "H1DrawHanoiDrawStroopSpeed Btw Exp and Control")


# ExpH1DrawHanoiDrawMath to ControlH1DrawHanoiDrawMath
H1DrawHanoiDrawMathResumption = statisticize.ttest(ExpH1DrawHanoiDrawMathDiffResumption,
                                              ControlH1DrawHanoiDrawMathDiffResumption, paired=False, alternative="greater")
H1DrawHanoiDrawMathResumption.insert(0, "Name", "H1DrawHanoiDrawMathResumption Btw Exp and Control")

H1DrawHanoiDrawMathInterruption = statisticize.ttest(ExpH1DrawHanoiDrawMathDiffInterruption,
                                              ControlH1DrawHanoiDrawMathDiffInterruption, paired=False, alternative="greater")
H1DrawHanoiDrawMathInterruption.insert(0, "Name", "H1DrawHanoiDrawMathInterruption Btw Exp and Control")

H1DrawHanoiDrawMathAccuracy = statisticize.ttest(ExpH1DrawHanoiDrawMathDiffAccuracy,
                                              ControlH1DrawHanoiDrawMathDiffAccuracy, paired=False, alternative="greater")
H1DrawHanoiDrawMathAccuracy.insert(0, "Name", "H1DrawHanoiDrawMathAccuracy Btw Exp and Control")

H1DrawHanoiDrawMathSpeed = statisticize.ttest(ExpH1DrawHanoiDrawMathDiffSpeed,
                                              ControlH1DrawHanoiDrawMathDiffSpeed, paired=False, alternative="greater")
H1DrawHanoiDrawMathSpeed.insert(0, "Name", "H1DrawHanoiDrawMathSpeed Btw Exp and Control")


# ExpH1HanoiDrawHanoiStroop to ControlH1HanoiDrawHanoiStroop
H1HanoiDrawHanoiStroopResumption = statisticize.ttest(ExpH1HanoiDrawHanoiStroopDiffResumption,
                                              ControlH1HanoiDrawHanoiStroopDiffResumption, paired=False, alternative="greater")
H1HanoiDrawHanoiStroopResumption.insert(0, "Name", "H1HanoiDrawHanoiStroopResumption Btw Exp and Control")

H1HanoiDrawHanoiStroopInterruption = statisticize.ttest(ExpH1HanoiDrawHanoiStroopDiffInterruption,
                                              ControlH1HanoiDrawHanoiStroopDiffInterruption, paired=False, alternative="greater")
H1HanoiDrawHanoiStroopInterruption.insert(0, "Name", "H1HanoiDrawHanoiStroopInterruption Btw Exp and Control")

H1HanoiDrawHanoiStroopAccuracy = statisticize.ttest(ExpH1HanoiDrawHanoiStroopDiffAccuracy,
                                              ControlH1HanoiDrawHanoiStroopDiffAccuracy, paired=False, alternative="greater")
H1HanoiDrawHanoiStroopAccuracy.insert(0, "Name", "H1HanoiDrawHanoiStroopAccuracy Btw Exp and Control")

H1HanoiDrawHanoiStroopSpeed = statisticize.ttest(ExpH1HanoiDrawHanoiStroopDiffSpeed,
                                              ControlH1HanoiDrawHanoiStroopDiffSpeed, paired=False, alternative="greater")
H1HanoiDrawHanoiStroopSpeed.insert(0, "Name", "H1HanoiDrawHanoiStroopSpeed Btw Exp and Control")


# ExpH1HanoiDrawHanoiMath to ControlH1HanoiDrawHanoiMath
H1HanoiDrawHanoiMathResumption = statisticize.ttest(ExpH1HanoiDrawHanoiMathDiffResumption,
                                              ControlH1HanoiDrawHanoiMathDiffResumption, paired=False, alternative="greater")
H1HanoiDrawHanoiMathResumption.insert(0, "Name", "H1HanoiDrawHanoiMathResumption Btw Exp and Control")

H1HanoiDrawHanoiMathInterruption = statisticize.ttest(ExpH1HanoiDrawHanoiMathDiffInterruption,
                                              ControlH1HanoiDrawHanoiMathDiffInterruption, paired=False, alternative="greater")
H1HanoiDrawHanoiMathInterruption.insert(0, "Name", "H1HanoiDrawHanoiMathInterruption Btw Exp and Control")

H1HanoiDrawHanoiMathAccuracy = statisticize.ttest(ExpH1HanoiDrawHanoiMathDiffAccuracy,
                                              ControlH1HanoiDrawHanoiMathDiffAccuracy, paired=False, alternative="greater")
H1HanoiDrawHanoiMathAccuracy.insert(0, "Name", "H1HanoiDrawHanoiMathAccuracy Btw Exp and Control")

H1HanoiDrawHanoiMathSpeed = statisticize.ttest(ExpH1HanoiDrawHanoiMathDiffSpeed,
                                              ControlH1HanoiDrawHanoiMathDiffSpeed, paired=False, alternative="greater")
H1HanoiDrawHanoiMathSpeed.insert(0, "Name", "H1HanoiDrawHanoiMathSpeed Btw Exp and Control")


# ExpH2StroopMathStroopDraw to ControlH2StroopMathStroopDraw
H2StroopMathStroopDrawResumption = statisticize.ttest(ExpH2StroopMathStroopDrawDiffResumption,
                                              ControlH2StroopMathStroopDrawDiffResumption, paired=False, alternative="less")
H2StroopMathStroopDrawResumption.insert(0, "Name", "H2StroopMathStroopDrawResumption Btw Exp and Control")

H2StroopMathStroopDrawInterruption = statisticize.ttest(ExpH2StroopMathStroopDrawDiffInterruption,
                                              ControlH2StroopMathStroopDrawDiffInterruption, paired=False, alternative="less")
H2StroopMathStroopDrawInterruption.insert(0, "Name", "H2StroopMathStroopDrawInterruption Btw Exp and Control")

H2StroopMathStroopDrawAccuracy = statisticize.ttest(ExpH2StroopMathStroopDrawDiffAccuracy,
                                              ControlH2StroopMathStroopDrawDiffAccuracy, paired=False, alternative="less")
H2StroopMathStroopDrawAccuracy.insert(0, "Name", "H2StroopMathStroopDrawAccuracy Btw Exp and Control")

H2StroopMathStroopDrawSpeed = statisticize.ttest(ExpH2StroopMathStroopDrawDiffSpeed,
                                              ControlH2StroopMathStroopDrawDiffSpeed, paired=False, alternative="less")
H2StroopMathStroopDrawSpeed.insert(0, "Name", "H2StroopMathStroopDrawSpeed Btw Exp and Control")


# ExpH2MathStroopMathDraw to ControlH2MathStroopMathDraw
H2MathStroopMathDrawResumption = statisticize.ttest(ExpH2MathStroopMathDrawDiffResumption,
                                              ControlH2MathStroopMathDrawDiffResumption, paired=False, alternative="less")
H2MathStroopMathDrawResumption.insert(0, "Name", "H2MathStroopMathDrawResumption Btw Exp and Control")

H2MathStroopMathDrawInterruption = statisticize.ttest(ExpH2MathStroopMathDrawDiffInterruption,
                                              ControlH2MathStroopMathDrawDiffInterruption, paired=False, alternative="less")
H2MathStroopMathDrawInterruption.insert(0, "Name", "H2MathStroopMathDrawInterruption Btw Exp and Control")

H2MathStroopMathDrawAccuracy = statisticize.ttest(ExpH2MathStroopMathDrawDiffAccuracy,
                                              ControlH2MathStroopMathDrawDiffAccuracy, paired=False, alternative="less")
H2MathStroopMathDrawAccuracy.insert(0, "Name", "H2MathStroopMathDrawAccuracy Btw Exp and Control")

H2MathStroopMathDrawSpeed = statisticize.ttest(ExpH2MathStroopMathDrawDiffSpeed,
                                              ControlH2MathStroopMathDrawDiffSpeed, paired=False, alternative="less")
H2MathStroopMathDrawSpeed.insert(0, "Name", "H2MathStroopMathDrawSpeed Btw Exp and Control")


# ExpH2StroopMathStroopHanoi to ControlH2StroopMathStroopHanoi
H2StroopMathStroopHanoiResumption = statisticize.ttest(ExpH2StroopMathStroopHanoiDiffResumption,
                                              ControlH2StroopMathStroopHanoiDiffResumption, paired=False, alternative="less")
H2StroopMathStroopHanoiResumption.insert(0, "Name", "H2StroopMathStroopHanoiResumption Btw Exp and Control")

H2StroopMathStroopHanoiInterruption = statisticize.ttest(ExpH2StroopMathStroopHanoiDiffInterruption,
                                              ControlH2StroopMathStroopHanoiDiffInterruption, paired=False, alternative="less")
H2StroopMathStroopHanoiInterruption.insert(0, "Name", "H2StroopMathStroopHanoiInterruption Btw Exp and Control")

H2StroopMathStroopHanoiAccuracy = statisticize.ttest(ExpH2StroopMathStroopHanoiDiffAccuracy,
                                              ControlH2StroopMathStroopHanoiDiffAccuracy, paired=False, alternative="less")
H2StroopMathStroopHanoiAccuracy.insert(0, "Name", "H2StroopMathStroopHanoiAccuracy Btw Exp and Control")

H2StroopMathStroopHanoiSpeed = statisticize.ttest(ExpH2StroopMathStroopHanoiDiffSpeed,
                                              ControlH2StroopMathStroopHanoiDiffSpeed, paired=False, alternative="less")
H2StroopMathStroopHanoiSpeed.insert(0, "Name", "H2StroopMathStroopHanoiSpeed Btw Exp and Control")


# ExpH2MathStroopMathHanoi to ControlH2MathStroopMathHanoi
H2MathStroopMathHanoiResumption = statisticize.ttest(ExpH2MathStroopMathHanoiDiffResumption,
                                              ControlH2MathStroopMathHanoiDiffResumption, paired=False, alternative="less")
H2MathStroopMathHanoiResumption.insert(0, "Name", "H2MathStroopMathHanoiResumption Btw Exp and Control")

H2MathStroopMathHanoiInterruption = statisticize.ttest(ExpH2MathStroopMathHanoiDiffInterruption,
                                              ControlH2MathStroopMathHanoiDiffInterruption, paired=False, alternative="less")
H2MathStroopMathHanoiInterruption.insert(0, "Name", "H2MathStroopMathHanoiInterruption Btw Exp and Control")

H2MathStroopMathHanoiAccuracy = statisticize.ttest(ExpH2MathStroopMathHanoiDiffAccuracy,
                                              ControlH2MathStroopMathHanoiDiffAccuracy, paired=False, alternative="less")
H2MathStroopMathHanoiAccuracy.insert(0, "Name", "H2MathStroopMathHanoiAccuracy Btw Exp and Control")

H2MathStroopMathHanoiSpeed = statisticize.ttest(ExpH2MathStroopMathHanoiDiffSpeed,
                                              ControlH2MathStroopMathHanoiDiffSpeed, paired=False, alternative="less")
H2MathStroopMathHanoiSpeed.insert(0, "Name", "H2MathStroopMathHanoiSpeed Btw Exp and Control")


# Concatenating the DataFrames containing metrics' stats
MetricsByH1AcrossConditions = pd.concat([H1DrawHanoiDrawStroopResumption,
                                         H1DrawHanoiDrawMathResumption,
                                         H1HanoiDrawHanoiStroopResumption,
                                         H1HanoiDrawHanoiMathResumption,
                                         H1DrawHanoiDrawStroopInterruption,
                                         H1DrawHanoiDrawMathInterruption,
                                         H1HanoiDrawHanoiStroopInterruption,
                                         H1HanoiDrawHanoiMathInterruption,
                                         H1DrawHanoiDrawStroopAccuracy,
                                         H1DrawHanoiDrawMathAccuracy,
                                         H1HanoiDrawHanoiStroopAccuracy,
                                         H1HanoiDrawHanoiMathAccuracy,
                                         H1DrawHanoiDrawStroopSpeed,
                                         H1DrawHanoiDrawMathSpeed,
                                         H1HanoiDrawHanoiStroopSpeed,
                                         H1HanoiDrawHanoiMathSpeed])
filenameForStats = "Metrics By H1 Btw Comparable Variations of Both Conditions"
MetricsByH1AcrossConditions.to_csv('../DataResults/Stats/InterVariationalInterConditionalStats/' + filenameForStats + '.csv')

MetricsByH2AcrossConditions = pd.concat([H2StroopMathStroopDrawResumption,
                                         H2MathStroopMathDrawResumption,
                                         H2StroopMathStroopHanoiResumption,
                                         H2MathStroopMathHanoiResumption,
                                         H2StroopMathStroopDrawInterruption,
                                         H2MathStroopMathDrawInterruption,
                                         H2StroopMathStroopHanoiInterruption,
                                         H2MathStroopMathHanoiInterruption,
                                         H2StroopMathStroopDrawAccuracy,
                                         H2MathStroopMathDrawAccuracy,
                                         H2StroopMathStroopHanoiAccuracy,
                                         H2MathStroopMathHanoiAccuracy,
                                         H2StroopMathStroopDrawSpeed,
                                         H2MathStroopMathDrawSpeed,
                                         H2StroopMathStroopHanoiSpeed,
                                         H2MathStroopMathHanoiSpeed])
filenameForStats = "Metrics By H2 Btw Comparable Variations of Both Conditions"
MetricsByH2AcrossConditions.to_csv('../DataResults/Stats/InterVariationalInterConditionalStats/' + filenameForStats + '.csv')


# Appending all values for each metric by hypothesis and by condition
# H1ExperimentalConditionResumptions is for all Resumption lags in first quadrant
H1ExperimentalConditionResumptions = ExpH1DrawHanoiDrawStroopDiffResumption + \
                                     ExpH1DrawHanoiDrawMathDiffResumption + \
                                     ExpH1HanoiDrawHanoiStroopDiffResumption + \
                                     ExpH1HanoiDrawHanoiMathDiffResumption
# H1ExperimentalConditionInterruptions is for all Interruption lags in first quadrant
H1ExperimentalConditionInterruptions = ExpH1DrawHanoiDrawStroopDiffInterruption + \
                                       ExpH1DrawHanoiDrawMathDiffInterruption + \
                                       ExpH1HanoiDrawHanoiStroopDiffInterruption + \
                                       ExpH1HanoiDrawHanoiMathDiffInterruption
# H1ExperimentalConditionAccuracies is for all Accuracies in first quadrant
H1ExperimentalConditionAccuracies = ExpH1DrawHanoiDrawStroopDiffAccuracy + \
                                    ExpH1DrawHanoiDrawMathDiffAccuracy + \
                                    ExpH1HanoiDrawHanoiStroopDiffAccuracy + \
                                    ExpH1HanoiDrawHanoiMathDiffAccuracy
# H1ExperimentalConditionSpeeds is for all Speeds in first quadrant
H1ExperimentalConditionSpeeds = ExpH1DrawHanoiDrawStroopDiffSpeed + \
                                ExpH1DrawHanoiDrawMathDiffSpeed + \
                                ExpH1HanoiDrawHanoiStroopDiffSpeed + \
                                ExpH1HanoiDrawHanoiMathDiffSpeed
# H1ControlConditionResumptions is for all Resumption lags in second quadrant (Clockwise)
H1ControlConditionResumptions = ControlH1DrawHanoiDrawStroopDiffResumption + \
                                ControlH1DrawHanoiDrawMathDiffResumption + \
                                ControlH1HanoiDrawHanoiStroopDiffResumption + \
                                ControlH1HanoiDrawHanoiMathDiffResumption
# H1ControlConditionInterruptions is for all Interruption lags in second quadrant (Clockwise)
H1ControlConditionInterruptions = ControlH1DrawHanoiDrawStroopDiffInterruption + \
                                  ControlH1DrawHanoiDrawMathDiffInterruption + \
                                  ControlH1HanoiDrawHanoiStroopDiffInterruption + \
                                  ControlH1HanoiDrawHanoiMathDiffInterruption
# H1ControlConditionAccuracies is for all Accuracy in second quadrant (Clockwise)
H1ControlConditionAccuracies = ControlH1DrawHanoiDrawStroopDiffAccuracy + \
                               ControlH1DrawHanoiDrawMathDiffAccuracy + \
                               ControlH1HanoiDrawHanoiStroopDiffAccuracy + \
                               ControlH1HanoiDrawHanoiMathDiffAccuracy
# H1ControlConditionSpeeds is for all Speed in second quadrant (Clockwise)
H1ControlConditionSpeeds = ControlH1DrawHanoiDrawStroopDiffSpeed + \
                           ControlH1DrawHanoiDrawMathDiffSpeed + \
                           ControlH1HanoiDrawHanoiStroopDiffSpeed + \
                           ControlH1HanoiDrawHanoiMathDiffSpeed
# Demarcation between H1 and H2
# H2ExperimentalConditionResumptions is for all Speed in fourth quadrant (Clockwise)
H2ExperimentalConditionResumptions = ExpH2StroopMathStroopDrawDiffResumption + \
                                     ExpH2MathStroopMathDrawDiffResumption + \
                                     ExpH2StroopMathStroopHanoiDiffResumption + \
                                     ExpH2MathStroopMathHanoiDiffResumption
H2ExperimentalConditionInterruptions = ExpH2StroopMathStroopDrawDiffInterruption + \
                                       ExpH2MathStroopMathDrawDiffInterruption + \
                                       ExpH2StroopMathStroopHanoiDiffInterruption + \
                                       ExpH2MathStroopMathHanoiDiffInterruption
H2ExperimentalConditionAccuracies = ExpH2StroopMathStroopDrawDiffAccuracy + \
                                    ExpH2MathStroopMathDrawDiffAccuracy + \
                                    ExpH2StroopMathStroopHanoiDiffAccuracy + \
                                    ExpH2MathStroopMathHanoiDiffAccuracy
H2ExperimentalConditionSpeeds = ExpH2StroopMathStroopDrawDiffSpeed + \
                                ExpH2MathStroopMathDrawDiffSpeed + \
                                ExpH2StroopMathStroopHanoiDiffSpeed + \
                                ExpH2MathStroopMathHanoiDiffSpeed
H2ControlConditionResumptions = ControlH2StroopMathStroopDrawDiffResumption + \
                                ControlH2MathStroopMathDrawDiffResumption + \
                                ControlH2StroopMathStroopHanoiDiffResumption + \
                                ControlH2MathStroopMathHanoiDiffResumption
H2ControlConditionInterruptions = ControlH2StroopMathStroopDrawDiffInterruption + \
                                  ControlH2MathStroopMathDrawDiffInterruption + \
                                  ControlH2StroopMathStroopHanoiDiffInterruption + \
                                  ControlH2MathStroopMathHanoiDiffInterruption
H2ControlConditionAccuracies = ControlH2StroopMathStroopDrawDiffAccuracy + \
                               ControlH2MathStroopMathDrawDiffAccuracy + \
                               ControlH2StroopMathStroopHanoiDiffAccuracy + \
                               ControlH2MathStroopMathHanoiDiffAccuracy
H2ControlConditionSpeeds = ControlH2StroopMathStroopDrawDiffSpeed + \
                           ControlH2MathStroopMathDrawDiffSpeed + \
                           ControlH2StroopMathStroopHanoiDiffSpeed + \
                           ControlH2MathStroopMathHanoiDiffSpeed

H1InterConditionsResumptionStats = statisticize.ttest(H1ExperimentalConditionResumptions,
                                              H1ControlConditionResumptions, paired=False, alternative="greater")

meanH1ExperimentalConditionResumptions = mean(H1ExperimentalConditionResumptions)
pstdevH1ExperimentalConditionResumptions = pstdev(H1ExperimentalConditionResumptions)

meanH1ControlConditionResumptions = mean(H1ControlConditionResumptions)
pstdevH1ControlConditionResumptions = pstdev(H1ControlConditionResumptions)

differenceBtwExpControlResumeH1 = meanH1ExperimentalConditionResumptions - meanH1ControlConditionResumptions

H1InterConditionsResumptionStats.insert(0, "Name", "Resumption Lag's Stats for H1 Btw Exp and Control")
H1InterConditionsResumptionStats.insert(9, "Mean Sample 1 (Seconds)", meanH1ExperimentalConditionResumptions)
H1InterConditionsResumptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevH1ExperimentalConditionResumptions)
H1InterConditionsResumptionStats.insert(11, "Mean Sample 2 (Seconds)", meanH1ControlConditionResumptions)
H1InterConditionsResumptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevH1ControlConditionResumptions)
H1InterConditionsResumptionStats.insert(13, "Diff In Seconds", differenceBtwExpControlResumeH1)


H1InterConditionsInterruptionStats = statisticize.ttest(H1ExperimentalConditionInterruptions,
                                              H1ControlConditionInterruptions, paired=False, alternative="greater")

meanH1ExperimentalConditionInterruptions = mean(H1ExperimentalConditionInterruptions)
pstdevH1ExperimentalConditionInterruptions = pstdev(H1ExperimentalConditionInterruptions)

meanH1ControlConditionInterruptions = mean(H1ControlConditionInterruptions)
pstdevH1ControlConditionInterruptions = pstdev(H1ControlConditionInterruptions)

differenceBtwExpControlAttendH1 = meanH1ExperimentalConditionInterruptions - meanH1ControlConditionInterruptions

H1InterConditionsInterruptionStats.insert(0, "Name", "Interruption Lag's Stats for H1 Btw Exp and Control")
H1InterConditionsInterruptionStats.insert(9, "Mean Sample 1 (Seconds)", meanH1ExperimentalConditionInterruptions)
H1InterConditionsInterruptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevH1ExperimentalConditionInterruptions)
H1InterConditionsInterruptionStats.insert(11, "Mean Sample 2 (Seconds)", meanH1ControlConditionInterruptions)
H1InterConditionsInterruptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevH1ControlConditionInterruptions)
H1InterConditionsInterruptionStats.insert(13, "Diff In Seconds", differenceBtwExpControlAttendH1)


H1InterConditionsAccuracyStats = statisticize.ttest(H1ExperimentalConditionAccuracies,
                                              H1ControlConditionAccuracies, paired=False, alternative="greater")

meanH1ExperimentalConditionAccuracies = mean(H1ExperimentalConditionAccuracies)
pstdevH1ExperimentalConditionAccuracies = pstdev(H1ExperimentalConditionAccuracies)

meanH1ControlConditionAccuracies = mean(H1ControlConditionAccuracies)
pstdevH1ControlConditionAccuracies = pstdev(H1ControlConditionAccuracies)

differenceBtwExpControlAccuracyH1 = meanH1ExperimentalConditionAccuracies - meanH1ControlConditionAccuracies

H1InterConditionsAccuracyStats.insert(0, "Name", "Accuracy's Stats for H1 Btw Exp and Control")
H1InterConditionsAccuracyStats.insert(9, "Mean Sample 1 (Seconds)", meanH1ExperimentalConditionAccuracies)
H1InterConditionsAccuracyStats.insert(10, "SD Sample 1 (Seconds)", pstdevH1ExperimentalConditionAccuracies)
H1InterConditionsAccuracyStats.insert(11, "Mean Sample 2 (Seconds)", meanH1ControlConditionAccuracies)
H1InterConditionsAccuracyStats.insert(12, "SD Sample 2 (Seconds)", pstdevH1ControlConditionAccuracies)
H1InterConditionsAccuracyStats.insert(13, "Diff In Seconds", differenceBtwExpControlAccuracyH1)


H1InterConditionsSpeedStats = statisticize.ttest(H1ExperimentalConditionSpeeds,
                                              H1ControlConditionSpeeds, paired=False, alternative="greater")

meanH1ExperimentalConditionSpeeds = mean(H1ExperimentalConditionSpeeds)
pstdevH1ExperimentalConditionSpeeds = pstdev(H1ExperimentalConditionSpeeds)

meanH1ControlConditionSpeeds = mean(H1ControlConditionSpeeds)
pstdevH1ControlConditionSpeeds = pstdev(H1ControlConditionSpeeds)

differenceBtwExpControlSpeedH1 = meanH1ExperimentalConditionSpeeds - meanH1ControlConditionSpeeds

H1InterConditionsSpeedStats.insert(0, "Name", "Speed's Stats for H1 Btw Exp and Control")
H1InterConditionsSpeedStats.insert(9, "Mean Sample 1 (Seconds)", meanH1ExperimentalConditionSpeeds)
H1InterConditionsSpeedStats.insert(10, "SD Sample 1 (Seconds)", pstdevH1ExperimentalConditionSpeeds)
H1InterConditionsSpeedStats.insert(11, "Mean Sample 2 (Seconds)", meanH1ControlConditionSpeeds)
H1InterConditionsSpeedStats.insert(12, "SD Sample 2 (Seconds)", pstdevH1ControlConditionSpeeds)
H1InterConditionsSpeedStats.insert(13, "Diff In Seconds", differenceBtwExpControlSpeedH1)


H2InterConditionsResumptionStats = statisticize.ttest(H2ExperimentalConditionResumptions,
                                              H2ControlConditionResumptions, paired=False, alternative="less")

meanH2ExperimentalConditionResumptions = mean(H2ExperimentalConditionResumptions)
pstdevH2ExperimentalConditionResumptions = pstdev(H2ExperimentalConditionResumptions)

meanH2ControlConditionResumptions = mean(H2ControlConditionResumptions)
pstdevH2ControlConditionResumptions = pstdev(H2ControlConditionResumptions)

differenceBtwExpControlResumeH2 = meanH2ExperimentalConditionResumptions - meanH2ControlConditionResumptions

H2InterConditionsResumptionStats.insert(0, "Name", "Resumption Lag's Stats for H2 Btw Exp and Control")
H2InterConditionsResumptionStats.insert(9, "Mean Sample 1 (Seconds)", meanH2ExperimentalConditionResumptions)
H2InterConditionsResumptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevH2ExperimentalConditionResumptions)
H2InterConditionsResumptionStats.insert(11, "Mean Sample 2 (Seconds)", meanH2ControlConditionResumptions)
H2InterConditionsResumptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevH2ControlConditionResumptions)
H2InterConditionsResumptionStats.insert(13, "Diff In Seconds", differenceBtwExpControlResumeH2)


H2InterConditionsInterruptionStats = statisticize.ttest(H2ExperimentalConditionInterruptions,
                                              H2ControlConditionInterruptions, paired=False, alternative="less")

meanH2ExperimentalConditionInterruptions = mean(H2ExperimentalConditionInterruptions)
pstdevH2ExperimentalConditionInterruptions = pstdev(H2ExperimentalConditionInterruptions)

meanH2ControlConditionInterruptions = mean(H2ControlConditionInterruptions)
pstdevH2ControlConditionInterruptions = pstdev(H2ControlConditionInterruptions)

differenceBtwExpControlAttendH2 = meanH2ExperimentalConditionInterruptions - meanH2ControlConditionInterruptions

H2InterConditionsInterruptionStats.insert(0, "Name", "Interruption Lag's Stats for H2 Btw Exp and Control")
H2InterConditionsInterruptionStats.insert(9, "Mean Sample 1 (Seconds)", meanH2ExperimentalConditionInterruptions)
H2InterConditionsInterruptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevH2ExperimentalConditionInterruptions)
H2InterConditionsInterruptionStats.insert(11, "Mean Sample 2 (Seconds)", meanH2ControlConditionInterruptions)
H2InterConditionsInterruptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevH2ControlConditionInterruptions)
H2InterConditionsInterruptionStats.insert(13, "Diff In Seconds", differenceBtwExpControlAttendH2)


H2InterConditionsAccuracyStats = statisticize.ttest(H2ExperimentalConditionAccuracies,
                                              H2ControlConditionAccuracies, paired=False, alternative="less")

meanH2ExperimentalConditionAccuracies = mean(H2ExperimentalConditionAccuracies)
pstdevH2ExperimentalConditionAccuracies = pstdev(H2ExperimentalConditionAccuracies)

meanH2ControlConditionAccuracies = mean(H2ControlConditionAccuracies)
pstdevH2ControlConditionAccuracies = pstdev(H2ControlConditionAccuracies)

differenceBtwExpControlAccuracyH2 = meanH2ExperimentalConditionAccuracies - meanH2ControlConditionAccuracies

H2InterConditionsAccuracyStats.insert(0, "Name", "Accuracy's Stats for H2 Btw Exp and Control")
H2InterConditionsAccuracyStats.insert(9, "Mean Sample 1 (Seconds)", meanH2ExperimentalConditionAccuracies)
H2InterConditionsAccuracyStats.insert(10, "SD Sample 1 (Seconds)", pstdevH2ExperimentalConditionAccuracies)
H2InterConditionsAccuracyStats.insert(11, "Mean Sample 2 (Seconds)", meanH2ControlConditionAccuracies)
H2InterConditionsAccuracyStats.insert(12, "SD Sample 2 (Seconds)", pstdevH2ControlConditionAccuracies)
H2InterConditionsAccuracyStats.insert(13, "Diff In Seconds", differenceBtwExpControlAccuracyH2)


H2InterConditionsSpeedStats = statisticize.ttest(H2ExperimentalConditionSpeeds,
                                              H2ControlConditionSpeeds, paired=False, alternative="less")

meanH2ExperimentalConditionSpeeds = mean(H2ExperimentalConditionSpeeds)
pstdevH2ExperimentalConditionSpeeds = pstdev(H2ExperimentalConditionSpeeds)

meanH2ControlConditionSpeeds = mean(H2ControlConditionSpeeds)
pstdevH2ControlConditionSpeeds = pstdev(H2ControlConditionSpeeds)

differenceBtwExpControlSpeedH2 = meanH2ExperimentalConditionSpeeds - meanH2ControlConditionSpeeds

H2InterConditionsSpeedStats.insert(0, "Name", "Speed's Stats for H2 Btw Exp and Control")
H2InterConditionsSpeedStats.insert(9, "Mean Sample 1 (Seconds)", meanH2ExperimentalConditionSpeeds)
H2InterConditionsSpeedStats.insert(10, "SD Sample 1 (Seconds)", pstdevH2ExperimentalConditionSpeeds)
H2InterConditionsSpeedStats.insert(11, "Mean Sample 2 (Seconds)", meanH2ControlConditionSpeeds)
H2InterConditionsSpeedStats.insert(12, "SD Sample 2 (Seconds)", pstdevH2ControlConditionSpeeds)
H2InterConditionsSpeedStats.insert(13, "Diff In Seconds", differenceBtwExpControlSpeedH2)

# Plotting of InterCondition Resumption Stats
plotTitle = 'Experimental Intervention vs. Control Comparison: H1'
yAxisLabel = 'Average Interruption Lag Times (Seconds)'
filenameForPlots = "ExperimentalControlH1Interruption"
interConditionPlotter(averageAttentionsLagListExpH1,
                      averageAttentionsLagListExpH1,
                      averageAttentionsLagListControlH1,
                      averageAttentionsLagListControlH1,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      H1InterConditionsInterruptionStats)

plotTitle = 'Experimental Intervention vs. Control Comparison: H1'
yAxisLabel = 'Average  Resumption Lag Times (Seconds)'
filenameForPlots = "ExperimentalControlH1Resumption"
interConditionPlotter(averageResumptionLagListExpH1,
                      averageResumptionLagListExpH1,
                      averageResumptionLagListControlH1,
                      averageResumptionLagListControlH1,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      H1InterConditionsResumptionStats)

plotTitle = 'Experimental Intervention vs. Control Comparison: H2'
yAxisLabel = 'Average Interruption Lag Times (Seconds)'
filenameForPlots = "ExperimentalControlH2Interruption"
interConditionPlotter(averageAttentionsLagListExpH2,
                      averageAttentionsLagListExpH2,
                      averageAttentionsLagListControlH2,
                      averageAttentionsLagListControlH2,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      H2InterConditionsInterruptionStats)

plotTitle = 'Experimental Intervention vs. Control Comparison: H2'
yAxisLabel = 'Average  Resumption Lag Times (Seconds)'
filenameForPlots = "ExperimentalControlH2Resumption"
interConditionPlotter(averageResumptionLagListExpH2,
                      averageResumptionLagListExpH2,
                      averageResumptionLagListControlH2,
                      averageResumptionLagListControlH2,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      H2InterConditionsResumptionStats)


# Concatenating the DataFrames containing metrics' stats
MetricsBothHypothesesByCondition = pd.concat([H1InterConditionsResumptionStats,
                                              H2InterConditionsResumptionStats,
                                              H1InterConditionsInterruptionStats,
                                              H2InterConditionsInterruptionStats,
                                              H1InterConditionsAccuracyStats,
                                              H2InterConditionsAccuracyStats,
                                              H1InterConditionsSpeedStats,
                                              H2InterConditionsSpeedStats])
filenameForStats = "Metrics for Each Hypothesis Btw Both Conditions"
MetricsBothHypothesesByCondition.to_csv('../DataResults/Stats/SummarizingStats/' + filenameForStats + '.csv')


# Appending all values for each metric by condition
ExperimentalConditionResumptions = H1ExperimentalConditionResumptions + H2ExperimentalConditionResumptions
ExperimentalConditionInterruptions = H1ExperimentalConditionInterruptions + H2ExperimentalConditionInterruptions
ExperimentalConditionAccuracies = H1ExperimentalConditionAccuracies + H2ExperimentalConditionAccuracies
ExperimentalConditionSpeeds = H1ExperimentalConditionSpeeds + H2ExperimentalConditionSpeeds

ControlConditionResumptions = H1ControlConditionResumptions + H2ControlConditionResumptions
ControlConditionInterruptions = H1ControlConditionInterruptions + H2ControlConditionInterruptions
ControlConditionAccuracies = H1ControlConditionAccuracies + H2ControlConditionAccuracies
ControlConditionSpeeds = H1ControlConditionSpeeds + H2ControlConditionSpeeds


InterConditionsResumptionStats = statisticize.ttest(ExperimentalConditionResumptions,
                                              ControlConditionResumptions, paired=False)#, alternative="less")

meanExperimentalConditionResumptions = mean(ExperimentalConditionResumptions)
pstdevExperimentalConditionResumptions = pstdev(ExperimentalConditionResumptions)

meanControlConditionResumptions = mean(ControlConditionResumptions)
pstdevControlConditionResumptions = pstdev(ControlConditionResumptions)

differenceBtwExpControlResume = meanExperimentalConditionResumptions - meanControlConditionResumptions
fractionalDifferenceBtwExpControlResume = differenceBtwExpControlResume/meanExperimentalConditionResumptions

InterConditionsResumptionStats.insert(0, "Name", "Resumption Lag's Stats Btw Exp and Control")
InterConditionsResumptionStats.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalConditionResumptions)
InterConditionsResumptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalConditionResumptions)
InterConditionsResumptionStats.insert(11, "Mean Sample 2 (Seconds)", meanControlConditionResumptions)
InterConditionsResumptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevControlConditionResumptions)
InterConditionsResumptionStats.insert(13, "Diff In Seconds", differenceBtwExpControlResume)
InterConditionsResumptionStats.insert(14, "Fraction In Seconds", fractionalDifferenceBtwExpControlResume)


InterConditionsInterruptionStats = statisticize.ttest(ExperimentalConditionInterruptions,
                                              ControlConditionInterruptions, paired=False)#, alternative="greater")

meanExperimentalConditionInterruptions = mean(ExperimentalConditionInterruptions)
pstdevExperimentalConditionInterruptions = pstdev(ExperimentalConditionInterruptions)

meanControlConditionInterruptions = mean(ControlConditionInterruptions)
pstdevControlConditionInterruptions = pstdev(ControlConditionInterruptions)

differenceBtwExpControlAttend = meanExperimentalConditionInterruptions - meanControlConditionInterruptions
fractionaldifferenceBtwExpControlAttend = differenceBtwExpControlAttend/meanExperimentalConditionInterruptions

InterConditionsInterruptionStats.insert(0, "Name", "Interruption Lag's Stats Btw Exp and Control")
InterConditionsInterruptionStats.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalConditionInterruptions)
InterConditionsInterruptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalConditionInterruptions)
InterConditionsInterruptionStats.insert(11, "Mean Sample 2 (Seconds)", meanControlConditionInterruptions)
InterConditionsInterruptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevControlConditionInterruptions)
InterConditionsInterruptionStats.insert(13, "Diff In Seconds", differenceBtwExpControlAttend)
InterConditionsInterruptionStats.insert(14, "Fraction In Seconds", fractionaldifferenceBtwExpControlAttend)


InterConditionsAccuracyStats = statisticize.ttest(ExperimentalConditionAccuracies,
                                              ControlConditionAccuracies, paired=False)#, alternative="greater")

meanExperimentalConditionAccuracies = mean(ExperimentalConditionAccuracies)
pstdevExperimentalConditionAccuracies = pstdev(ExperimentalConditionAccuracies)

meanControlConditionAccuracies = mean(ControlConditionAccuracies)
pstdevControlConditionAccuracies = pstdev(ControlConditionAccuracies)

differenceBtwExpControlAccuracy = meanExperimentalConditionAccuracies - meanControlConditionAccuracies
fractionaldifferenceBtwExpControlAccuracy = differenceBtwExpControlAccuracy

InterConditionsAccuracyStats.insert(0, "Name", "Accuracy's Stats Btw Exp and Control")
InterConditionsAccuracyStats.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalConditionAccuracies)
InterConditionsAccuracyStats.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalConditionAccuracies)
InterConditionsAccuracyStats.insert(11, "Mean Sample 2 (Seconds)", meanControlConditionAccuracies)
InterConditionsAccuracyStats.insert(12, "SD Sample 2 (Seconds)", pstdevControlConditionAccuracies)
InterConditionsAccuracyStats.insert(13, "Diff In Seconds", differenceBtwExpControlAccuracy)
InterConditionsAccuracyStats.insert(14, "Fraction In Seconds", fractionaldifferenceBtwExpControlAccuracy)


InterConditionsSpeedStats = statisticize.ttest(ExperimentalConditionSpeeds,
                                              ControlConditionSpeeds, paired=False)#, alternative="greater")

meanExperimentalConditionSpeeds = mean(ExperimentalConditionSpeeds)
pstdevExperimentalConditionSpeeds = pstdev(ExperimentalConditionSpeeds)

meanControlConditionSpeeds = mean(ControlConditionSpeeds)
pstdevControlConditionSpeeds = pstdev(ControlConditionSpeeds)

differenceBtwExpControlSpeed = meanExperimentalConditionSpeeds - meanControlConditionSpeeds
fractionaldifferenceBtwExpControlSpeed = differenceBtwExpControlSpeed/meanExperimentalConditionSpeeds

InterConditionsSpeedStats.insert(0, "Name", "Speed's Stats Btw Exp and Control")
InterConditionsSpeedStats.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalConditionSpeeds)
InterConditionsSpeedStats.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalConditionSpeeds)
InterConditionsSpeedStats.insert(11, "Mean Sample 2 (Seconds)", meanControlConditionSpeeds)
InterConditionsSpeedStats.insert(12, "SD Sample 2 (Seconds)", pstdevControlConditionSpeeds)
InterConditionsSpeedStats.insert(13, "Diff In Seconds", differenceBtwExpControlSpeed)
InterConditionsSpeedStats.insert(14, "Fraction In Seconds", fractionaldifferenceBtwExpControlSpeed)


space = [""]
InterConditionMetricsStatsHeader = pd.DataFrame(data={"": space})
InterConditionMetricsStatsHeader.insert(0, "Name", "Metrics Stats By Entire Condition Btw Exp and Control (H1 + H2 in Control), (H1 + H2 in Exp)")#"Metrics Stats By Entire Condition Btw Exp and Control")

# Concatenating the DataFrames containing metrics' stats
MetricsStatsByEntireCondition = pd.concat([InterConditionsResumptionStats,
                                           InterConditionsInterruptionStats,
                                           InterConditionsAccuracyStats,
                                           InterConditionsSpeedStats])
filenameForStats = "Metrics Stats By Entire Condition Btw Exp and Control"
# MetricsStatsByEntireCondition.to_csv('../DataResults/Stats/SummarizingStats/' + filenameForStats + '.csv')


# Appending all values for each metric by hypothesis across conditions

IntraHypothesisOneInterConditionResumptions = H1ExperimentalConditionResumptions + H1ControlConditionResumptions
IntraHypothesisOneInterConditionInterruptions = H1ExperimentalConditionInterruptions + H1ControlConditionInterruptions
IntraHypothesisOneInterConditionAccuracies = H1ExperimentalConditionAccuracies + H1ControlConditionAccuracies
IntraHypothesisOneInterConditionSpeeds = H1ExperimentalConditionSpeeds + H1ControlConditionSpeeds

IntraHypothesisTwoInterConditionResumptions = H2ExperimentalConditionResumptions + H2ControlConditionResumptions
IntraHypothesisTwoInterConditionInterruptions = H2ExperimentalConditionInterruptions + H2ControlConditionInterruptions
IntraHypothesisTwoInterConditionAccuracies = H2ExperimentalConditionAccuracies + H2ControlConditionAccuracies
IntraHypothesisTwoInterConditionSpeeds = H2ExperimentalConditionSpeeds + H2ControlConditionSpeeds

# Resumption Lag's Stats BTW H1 and H2 Across Conditions
IntraHypothesisInterConditionsResumptionStats = statisticize.ttest(IntraHypothesisOneInterConditionResumptions,
                                              IntraHypothesisTwoInterConditionResumptions, paired=False)#, alternative="less")

meanIntraHypothesisOneInterConditionResumptions = mean(IntraHypothesisOneInterConditionResumptions)
pstdevIntraHypothesisOneInterConditionResumptions = pstdev(IntraHypothesisOneInterConditionResumptions)

meanIntraHypothesisTwoInterConditionResumptions = mean(IntraHypothesisTwoInterConditionResumptions)
pstdevIntraHypothesisTwoInterConditionResumptions = pstdev(IntraHypothesisTwoInterConditionResumptions)

differenceBtwH1andH2Resume = meanIntraHypothesisOneInterConditionResumptions - meanIntraHypothesisTwoInterConditionResumptions

IntraHypothesisInterConditionsResumptionStats.insert(0, "Name", "Resumption Lag's Stats BTW H1 and H2 Across Conditions")
IntraHypothesisInterConditionsResumptionStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneInterConditionResumptions)
IntraHypothesisInterConditionsResumptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneInterConditionResumptions)
IntraHypothesisInterConditionsResumptionStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoInterConditionResumptions)
IntraHypothesisInterConditionsResumptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoInterConditionResumptions)
IntraHypothesisInterConditionsResumptionStats.insert(13, "Diff In Seconds", differenceBtwH1andH2Resume)

# Interruption Lag's Stats BTW H1 and H2 Across Conditions
IntraHypothesisInterConditionsInterruptionStats = statisticize.ttest(IntraHypothesisOneInterConditionInterruptions,
                                              IntraHypothesisTwoInterConditionInterruptions, paired=False)#, alternative="greater")

meanIntraHypothesisOneInterConditionInterruptions = mean(IntraHypothesisOneInterConditionInterruptions)
pstdevIntraHypothesisOneInterConditionInterruptions = pstdev(IntraHypothesisOneInterConditionInterruptions)

meanIntraHypothesisTwoInterConditionInterruptions = mean(IntraHypothesisTwoInterConditionInterruptions)
pstdevIntraHypothesisTwoInterConditionInterruptions = pstdev(IntraHypothesisTwoInterConditionInterruptions)

differenceBtwH1andH2Attend = meanIntraHypothesisOneInterConditionInterruptions - meanIntraHypothesisTwoInterConditionInterruptions

IntraHypothesisInterConditionsInterruptionStats.insert(0, "Name", "Interruption Lag's Stats BTW H1 and H2 Across Conditions")
IntraHypothesisInterConditionsInterruptionStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneInterConditionInterruptions)
IntraHypothesisInterConditionsInterruptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneInterConditionInterruptions)
IntraHypothesisInterConditionsInterruptionStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoInterConditionInterruptions)
IntraHypothesisInterConditionsInterruptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoInterConditionInterruptions)
IntraHypothesisInterConditionsInterruptionStats.insert(13, "Diff In Seconds", differenceBtwH1andH2Attend)


# Accuracy's Stats BTW H1 and H2 Across Conditions
IntraHypothesisInterConditionsAccuraciesStats = statisticize.ttest(IntraHypothesisOneInterConditionAccuracies,
                                              IntraHypothesisTwoInterConditionAccuracies, paired=False)#, alternative="less")

meanIntraHypothesisOneInterConditionAccuracies = mean(IntraHypothesisOneInterConditionAccuracies)
pstdevIntraHypothesisOneInterConditionAccuracies = pstdev(IntraHypothesisOneInterConditionAccuracies)

meanIntraHypothesisTwoInterConditionAccuracies = mean(IntraHypothesisTwoInterConditionAccuracies)
pstdevIntraHypothesisTwoInterConditionAccuracies = pstdev(IntraHypothesisTwoInterConditionAccuracies)

differenceBtwH1andH2Accuracy = meanIntraHypothesisOneInterConditionAccuracies - meanIntraHypothesisTwoInterConditionAccuracies

IntraHypothesisInterConditionsAccuraciesStats.insert(0, "Name", "Accuracies Lag's Stats BTW H1 and H2 Across Conditions")
IntraHypothesisInterConditionsAccuraciesStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneInterConditionAccuracies)
IntraHypothesisInterConditionsAccuraciesStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneInterConditionAccuracies)
IntraHypothesisInterConditionsAccuraciesStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoInterConditionAccuracies)
IntraHypothesisInterConditionsAccuraciesStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoInterConditionAccuracies)
IntraHypothesisInterConditionsAccuraciesStats.insert(13, "Diff In Seconds", differenceBtwH1andH2Accuracy)


# Accuracy's Stats BTW H1 and H2 Across Conditions
IntraHypothesisInterConditionsSpeedsStats = statisticize.ttest(IntraHypothesisOneInterConditionSpeeds,
                                              IntraHypothesisTwoInterConditionSpeeds, paired=False)#, alternative="less")

meanIntraHypothesisOneInterConditionSpeeds = mean(IntraHypothesisOneInterConditionSpeeds)
pstdevIntraHypothesisOneInterConditionSpeeds = pstdev(IntraHypothesisOneInterConditionSpeeds)

meanIntraHypothesisTwoInterConditionSpeeds = mean(IntraHypothesisTwoInterConditionSpeeds)
pstdevIntraHypothesisTwoInterConditionSpeeds = pstdev(IntraHypothesisTwoInterConditionSpeeds)

differenceBtwH1andH2Speeds = meanIntraHypothesisOneInterConditionSpeeds - meanIntraHypothesisTwoInterConditionSpeeds

IntraHypothesisInterConditionsSpeedsStats.insert(0, "Name", "Speeds Lag's Stats BTW H1 and H2 Across Conditions")
IntraHypothesisInterConditionsSpeedsStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneInterConditionSpeeds)
IntraHypothesisInterConditionsSpeedsStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneInterConditionSpeeds)
IntraHypothesisInterConditionsSpeedsStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoInterConditionSpeeds)
IntraHypothesisInterConditionsSpeedsStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoInterConditionSpeeds)
IntraHypothesisInterConditionsSpeedsStats.insert(13, "Diff In Seconds", differenceBtwH1andH2Speeds)


space = [""]
intraHypothesisInterConditionMetricsStatsHeader = pd.DataFrame(data={"": space})
intraHypothesisInterConditionMetricsStatsHeader.insert(0, "Name", "Metrics Stats BTW H1 and H2 Across Conditions (H1 in Control + H1 in Exp), (H2 in Control + H2 in Exp)")#"Metrics Stats BTW H1 and H2 Across Conditions")

# Concatenating the DataFrames containing inter-hypothesis' metrics' stats
MetricsStatsBTWHypothesisByEntireCondition = pd.concat([IntraHypothesisInterConditionsResumptionStats,
                                                        IntraHypothesisInterConditionsInterruptionStats,
                                                        IntraHypothesisInterConditionsAccuraciesStats,
                                                        IntraHypothesisInterConditionsSpeedsStats])
filenameForStats = "Metrics Stats BTW H1 and H2 Across Conditions"
# MetricsStatsBTWHypothesisByEntireCondition.to_csv('../DataResults/Stats/SummarizingStats/' + filenameForStats + '.csv')



# Appending all values for each metric by hypothesis within conditions
# H1 Experimental
IntraHypothesisOneIntraExperimentalConditionResumptions = H1ExperimentalConditionResumptions
IntraHypothesisOneIntraExperimentalConditionInterruptions = H1ExperimentalConditionInterruptions
IntraHypothesisOneIntraExperimentalConditionAccuracies = H1ExperimentalConditionAccuracies
IntraHypothesisOneIntraExperimentalConditionSpeeds = H1ExperimentalConditionSpeeds

# H2 Experimental
IntraHypothesisTwoIntraExperimentalConditionResumptions = H2ExperimentalConditionResumptions
IntraHypothesisTwoIntraExperimentalConditionInterruptions = H2ExperimentalConditionInterruptions
IntraHypothesisTwoIntraExperimentalConditionAccuracies = H2ExperimentalConditionAccuracies
IntraHypothesisTwoIntraExperimentalConditionSpeeds = H2ExperimentalConditionSpeeds

# H1 Control
IntraHypothesisOneIntraControlConditionResumptions = H1ControlConditionResumptions
IntraHypothesisOneIntraControlConditionInterruptions = H1ControlConditionInterruptions
IntraHypothesisOneIntraControlConditionAccuracies = H1ControlConditionAccuracies
IntraHypothesisOneIntraControlConditionSpeeds = H1ControlConditionSpeeds

# H2 Control
IntraHypothesisTwoIntraControlConditionResumptions = H2ControlConditionResumptions
IntraHypothesisTwoIntraControlConditionInterruptions = H2ControlConditionInterruptions
IntraHypothesisTwoIntraControlConditionAccuracies = H2ControlConditionAccuracies
IntraHypothesisTwoIntraControlConditionSpeeds = H2ControlConditionSpeeds

# Resumption Lag's Stats BTW H1 and H2 within Experimentation
IntraHypothesisIntraExperimentationResumptionStats = statisticize.ttest(IntraHypothesisOneIntraExperimentalConditionResumptions,
                                              IntraHypothesisTwoIntraExperimentalConditionResumptions, paired=False)#, alternative="less")

meanIntraHypothesisOneIntraExperimentationResumptions = mean(IntraHypothesisOneIntraExperimentalConditionResumptions)
pstdevIntraHypothesisOneIntraExperimentationResumptions = pstdev(IntraHypothesisOneIntraExperimentalConditionResumptions)

meanIntraHypothesisTwoIntraExperimentationResumptions = mean(IntraHypothesisTwoIntraExperimentalConditionResumptions)
pstdevIntraHypothesisTwoIntraExperimentationResumptions = pstdev(IntraHypothesisTwoIntraExperimentalConditionResumptions)

differenceBtwH1andH2ExperimentationResume = meanIntraHypothesisOneIntraExperimentationResumptions - meanIntraHypothesisTwoIntraExperimentationResumptions

IntraHypothesisIntraExperimentationResumptionStats.insert(0, "Name", "Resumption Lag's Stats BTW H1 and H2 Experimental Intervention")
IntraHypothesisIntraExperimentationResumptionStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationResumptions)
IntraHypothesisIntraExperimentationResumptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationResumptions)
IntraHypothesisIntraExperimentationResumptionStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraExperimentationResumptions)
IntraHypothesisIntraExperimentationResumptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraExperimentationResumptions)
IntraHypothesisIntraExperimentationResumptionStats.insert(13, "Diff In Seconds", differenceBtwH1andH2ExperimentationResume)

# Interruption Lag's Stats BTW H1 and H2 within Conditions
IntraHypothesisIntraExperimentationInterruptionStats = statisticize.ttest(IntraHypothesisOneIntraExperimentalConditionInterruptions,
                                              IntraHypothesisTwoIntraExperimentalConditionInterruptions, paired=False, alternative="greater")

meanIntraHypothesisOneIntraExperimentationInterruptions = mean(IntraHypothesisOneIntraExperimentalConditionInterruptions)
pstdevIntraHypothesisOneIntraExperimentationInterruptions = pstdev(IntraHypothesisOneIntraExperimentalConditionInterruptions)

meanIntraHypothesisTwoIntraExperimentationInterruptions = mean(IntraHypothesisTwoIntraExperimentalConditionInterruptions)
pstdevIntraHypothesisTwoIntraExperimentationInterruptions = pstdev(IntraHypothesisTwoIntraExperimentalConditionInterruptions)

differenceBtwH1andH2ExperimentationAttend = meanIntraHypothesisOneIntraExperimentationInterruptions - meanIntraHypothesisTwoIntraExperimentationInterruptions

IntraHypothesisIntraExperimentationInterruptionStats.insert(0, "Name", "Interruption Lag's Stats BTW H1 and H2 Experimental Intervention")
IntraHypothesisIntraExperimentationInterruptionStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationInterruptions)
IntraHypothesisIntraExperimentationInterruptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationInterruptions)
IntraHypothesisIntraExperimentationInterruptionStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraExperimentationInterruptions)
IntraHypothesisIntraExperimentationInterruptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraExperimentationInterruptions)
IntraHypothesisIntraExperimentationInterruptionStats.insert(13, "Diff In Seconds", differenceBtwH1andH2ExperimentationAttend)

# Accuracy's Stats BTW H1 and H2 within Conditions
IntraHypothesisIntraExperimentationAccuraciesStats = statisticize.ttest(IntraHypothesisOneIntraExperimentalConditionAccuracies,
                                              IntraHypothesisTwoIntraExperimentalConditionAccuracies, paired=False, alternative="greater")

meanIntraHypothesisOneIntraExperimentationAccuracies = mean(IntraHypothesisOneIntraExperimentalConditionAccuracies)
pstdevIntraHypothesisOneIntraExperimentationAccuracies = pstdev(IntraHypothesisOneIntraExperimentalConditionAccuracies)

meanIntraHypothesisTwoIntraExperimentationAccuracies = mean(IntraHypothesisTwoIntraExperimentalConditionAccuracies)
pstdevIntraHypothesisTwoIntraExperimentationAccuracies = pstdev(IntraHypothesisTwoIntraExperimentalConditionAccuracies)

differenceBtwH1andH2ExperimentationAccuracy = meanIntraHypothesisOneIntraExperimentationAccuracies - meanIntraHypothesisTwoIntraExperimentationAccuracies

IntraHypothesisIntraExperimentationAccuraciesStats.insert(0, "Name", "Accuracies Lag's Stats BTW H1 and H2 Experimental Intervention")
IntraHypothesisIntraExperimentationAccuraciesStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationAccuracies)
IntraHypothesisIntraExperimentationAccuraciesStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationAccuracies)
IntraHypothesisIntraExperimentationAccuraciesStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraExperimentationAccuracies)
IntraHypothesisIntraExperimentationAccuraciesStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraExperimentationAccuracies)
IntraHypothesisIntraExperimentationAccuraciesStats.insert(13, "Diff In Seconds", differenceBtwH1andH2ExperimentationAccuracy)

# Accuracy's Stats BTW H1 and H2 within Conditions
IntraHypothesisIntraExperimentationSpeedsStats = statisticize.ttest(IntraHypothesisOneIntraExperimentalConditionSpeeds,
                                              IntraHypothesisTwoIntraExperimentalConditionSpeeds, paired=False)#, alternative="less")

meanIntraHypothesisOneIntraExperimentationSpeeds = mean(IntraHypothesisOneIntraExperimentalConditionSpeeds)
pstdevIntraHypothesisOneIntraExperimentationSpeeds = pstdev(IntraHypothesisOneIntraExperimentalConditionSpeeds)

meanIntraHypothesisTwoIntraExperimentationSpeeds = mean(IntraHypothesisTwoIntraExperimentalConditionSpeeds)
pstdevIntraHypothesisTwoIntraExperimentationSpeeds = pstdev(IntraHypothesisTwoIntraExperimentalConditionSpeeds)

differenceBtwH1andH2ExperimentationSpeeds = meanIntraHypothesisOneIntraExperimentationSpeeds - meanIntraHypothesisTwoIntraExperimentationSpeeds

IntraHypothesisIntraExperimentationSpeedsStats.insert(0, "Name", "Speeds Lag's Stats BTW H1 and H2 Experimental Intervention")
IntraHypothesisIntraExperimentationSpeedsStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationSpeeds)
IntraHypothesisIntraExperimentationSpeedsStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationSpeeds)
IntraHypothesisIntraExperimentationSpeedsStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraExperimentationSpeeds)
IntraHypothesisIntraExperimentationSpeedsStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraExperimentationSpeeds)
IntraHypothesisIntraExperimentationSpeedsStats.insert(13, "Diff In Seconds", differenceBtwH1andH2ExperimentationSpeeds)

space = [""]
intraHypothesisMetricsStatsExpHeader = pd.DataFrame(data={"": space})
intraHypothesisMetricsStatsExpHeader.insert(0, "Name", "Metrics Stats BTW H1 and H2 within Experimental Intervention")

# Concatenating the DataFrames containing inter-hypothesis' metrics' stats
MetricsStatsBTWHypothesisByExperimentalCondition = pd.concat([IntraHypothesisIntraExperimentationResumptionStats,
                                                              IntraHypothesisIntraExperimentationInterruptionStats,
                                                              IntraHypothesisIntraExperimentationAccuraciesStats,
                                                              IntraHypothesisIntraExperimentationSpeedsStats])
filenameForStats = "Metrics Stats BTW H1 and H2 within Experimental Intervention"
# MetricsStatsBTWHypothesisByExperimentalCondition.to_csv('../DataResults/Stats/SummarizingStats/' + filenameForStats + '.csv')



# Resumption Lag's Stats BTW H1 and H2 within Control Comparison
IntraHypothesisIntraControlResumptionStats = statisticize.ttest(IntraHypothesisOneIntraControlConditionResumptions,
                                              IntraHypothesisTwoIntraControlConditionResumptions, paired=False)#, alternative="less")

meanIntraHypothesisOneIntraControlResumptions = mean(IntraHypothesisOneIntraControlConditionResumptions)
pstdevIntraHypothesisOneIntraControlResumptions = pstdev(IntraHypothesisOneIntraControlConditionResumptions)

meanIntraHypothesisTwoIntraControlResumptions = mean(IntraHypothesisTwoIntraControlConditionResumptions)
pstdevIntraHypothesisTwoIntraControlResumptions = pstdev(IntraHypothesisTwoIntraControlConditionResumptions)

differenceBtwH1andH2ControlResume = meanIntraHypothesisOneIntraControlResumptions - meanIntraHypothesisTwoIntraControlResumptions

IntraHypothesisIntraControlResumptionStats.insert(0, "Name", "Resumption Lag's Stats BTW H1 and H2 Control Comparison")
IntraHypothesisIntraControlResumptionStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraControlResumptions)
IntraHypothesisIntraControlResumptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraControlResumptions)
IntraHypothesisIntraControlResumptionStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraControlResumptions)
IntraHypothesisIntraControlResumptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraControlResumptions)
IntraHypothesisIntraControlResumptionStats.insert(13, "Diff In Seconds", differenceBtwH1andH2ControlResume)

# Interruption Lag's Stats BTW H1 and H2 within Conditions
IntraHypothesisIntraControlInterruptionStats = statisticize.ttest(IntraHypothesisOneIntraControlConditionInterruptions,
                                              IntraHypothesisTwoIntraControlConditionInterruptions, paired=False)#, alternative="less")

meanIntraHypothesisOneIntraControlInterruptions = mean(IntraHypothesisOneIntraControlConditionInterruptions)
pstdevIntraHypothesisOneIntraControlInterruptions = pstdev(IntraHypothesisOneIntraControlConditionInterruptions)

meanIntraHypothesisTwoIntraControlInterruptions = mean(IntraHypothesisTwoIntraControlConditionInterruptions)
pstdevIntraHypothesisTwoIntraControlInterruptions = pstdev(IntraHypothesisTwoIntraControlConditionInterruptions)

differenceBtwH1andH2ControlAttend = meanIntraHypothesisOneIntraControlInterruptions - meanIntraHypothesisTwoIntraControlInterruptions

IntraHypothesisIntraControlInterruptionStats.insert(0, "Name", "Interruption Lag's Stats BTW H1 and H2 Control Comparison")
IntraHypothesisIntraControlInterruptionStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraControlInterruptions)
IntraHypothesisIntraControlInterruptionStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraControlInterruptions)
IntraHypothesisIntraControlInterruptionStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraControlInterruptions)
IntraHypothesisIntraControlInterruptionStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraControlInterruptions)
IntraHypothesisIntraControlInterruptionStats.insert(13, "Diff In Seconds", differenceBtwH1andH2ControlAttend)

# Accuracy's Stats BTW H1 and H2 within Conditions
IntraHypothesisIntraControlAccuraciesStats = statisticize.ttest(IntraHypothesisOneIntraControlConditionAccuracies,
                                              IntraHypothesisTwoIntraControlConditionAccuracies, paired=False, alternative="less")

meanIntraHypothesisOneIntraControlAccuracies = mean(IntraHypothesisOneIntraControlConditionAccuracies)
pstdevIntraHypothesisOneIntraControlAccuracies = pstdev(IntraHypothesisOneIntraControlConditionAccuracies)

meanIntraHypothesisTwoIntraControlAccuracies = mean(IntraHypothesisTwoIntraControlConditionAccuracies)
pstdevIntraHypothesisTwoIntraControlAccuracies = pstdev(IntraHypothesisTwoIntraControlConditionAccuracies)

differenceBtwH1andH2ControlAccuracy = meanIntraHypothesisOneIntraControlAccuracies - meanIntraHypothesisTwoIntraControlAccuracies

IntraHypothesisIntraControlAccuraciesStats.insert(0, "Name", "Accuracies Lag's Stats BTW H1 and H2 Control Comparison")
IntraHypothesisIntraControlAccuraciesStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraControlAccuracies)
IntraHypothesisIntraControlAccuraciesStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraControlAccuracies)
IntraHypothesisIntraControlAccuraciesStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraControlAccuracies)
IntraHypothesisIntraControlAccuraciesStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraControlAccuracies)
IntraHypothesisIntraControlAccuraciesStats.insert(13, "Diff In Seconds", differenceBtwH1andH2ControlAccuracy)

# Accuracy's Stats BTW H1 and H2 within Conditions
IntraHypothesisIntraControlSpeedsStats = statisticize.ttest(IntraHypothesisOneIntraControlConditionSpeeds,
                                              IntraHypothesisTwoIntraControlConditionSpeeds, paired=False)#, alternative="less")

meanIntraHypothesisOneIntraControlSpeeds = mean(IntraHypothesisOneIntraControlConditionSpeeds)
pstdevIntraHypothesisOneIntraControlSpeeds = pstdev(IntraHypothesisOneIntraControlConditionSpeeds)

meanIntraHypothesisTwoIntraControlSpeeds = mean(IntraHypothesisTwoIntraControlConditionSpeeds)
pstdevIntraHypothesisTwoIntraControlSpeeds = pstdev(IntraHypothesisTwoIntraControlConditionSpeeds)

differenceBtwH1andH2ControlSpeeds = meanIntraHypothesisOneIntraControlSpeeds - meanIntraHypothesisTwoIntraControlSpeeds

IntraHypothesisIntraControlSpeedsStats.insert(0, "Name", "Speeds Lag's Stats BTW H1 and H2 Control Comparison")
IntraHypothesisIntraControlSpeedsStats.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraControlSpeeds)
IntraHypothesisIntraControlSpeedsStats.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraControlSpeeds)
IntraHypothesisIntraControlSpeedsStats.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraControlSpeeds)
IntraHypothesisIntraControlSpeedsStats.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraControlSpeeds)
IntraHypothesisIntraControlSpeedsStats.insert(13, "Diff In Seconds", differenceBtwH1andH2ControlSpeeds)

space = [""]
intraHypothesisMetricsStatsControlHeader = pd.DataFrame(data={"": space})
intraHypothesisMetricsStatsControlHeader.insert(0, "Name", "Metrics Stats BTW H1 and H2 within Control Comparison")

# Concatenating the DataFrames containing intra-hypothesis' metrics' stats
MetricsStatsBTWHypothesisByControlCondition = pd.concat([IntraHypothesisIntraControlResumptionStats,
                                                        IntraHypothesisIntraControlInterruptionStats,
                                                        IntraHypothesisIntraControlAccuraciesStats,
                                                        IntraHypothesisIntraControlSpeedsStats])

filenameForStats = "Metrics Stats BTW H1 and H2 within Control Comparison"
# MetricsStatsBTWHypothesisByControlCondition.to_csv('../DataResults/Stats/SummarizingStats/' + filenameForStats + '.csv')



# Stats between conditions within same hypotheses

# Resumption Lag's Stats BTW Interventional H1 and Comparison H1
InterventionalH1andComparisonH1Resume = statisticize.ttest(IntraHypothesisOneIntraExperimentalConditionResumptions,
                                              IntraHypothesisOneIntraControlConditionResumptions, paired=False, alternative="greater")

meanIntraHypothesisOneIntraExperimentationResumptions = mean(IntraHypothesisOneIntraExperimentalConditionResumptions)
pstdevIntraHypothesisOneIntraExperimentationResumptions = pstdev(IntraHypothesisOneIntraExperimentalConditionResumptions)

meanIntraHypothesisOneIntraControlConditionResumptions = mean(IntraHypothesisOneIntraControlConditionResumptions)
pstdevIntraHypothesisOneIntraControlConditionResumptions = pstdev(IntraHypothesisOneIntraControlConditionResumptions)

differenceBtwH1EXPandH1CONResume = meanIntraHypothesisOneIntraExperimentationResumptions - meanIntraHypothesisOneIntraControlConditionResumptions
fractionalDifferenceBtwH1EXPandH1CONResume = differenceBtwH1EXPandH1CONResume/meanIntraHypothesisOneIntraExperimentationResumptions

InterventionalH1andComparisonH1Resume.insert(0, "Name", "Resumption Lag's Stats BTW Interventional H1 and Comparison H1")
InterventionalH1andComparisonH1Resume.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationResumptions)
InterventionalH1andComparisonH1Resume.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationResumptions)
InterventionalH1andComparisonH1Resume.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisOneIntraControlConditionResumptions)
InterventionalH1andComparisonH1Resume.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisOneIntraControlConditionResumptions)
InterventionalH1andComparisonH1Resume.insert(13, "Diff In Seconds", differenceBtwH1EXPandH1CONResume)
InterventionalH1andComparisonH1Resume.insert(14, "Fraction In Seconds", fractionalDifferenceBtwH1EXPandH1CONResume)

# Interruption Lag's Stats BTW Interventional H1 and Comparison H1
InterventionalH1andComparisonH1Attend = statisticize.ttest(IntraHypothesisOneIntraExperimentalConditionInterruptions,
                                              IntraHypothesisOneIntraControlConditionInterruptions, paired=False, alternative="greater")

meanIntraHypothesisOneIntraExperimentationInterruptions = mean(IntraHypothesisOneIntraExperimentalConditionInterruptions)
pstdevIntraHypothesisOneIntraExperimentationInterruptions = pstdev(IntraHypothesisOneIntraExperimentalConditionInterruptions)

meanIntraHypothesisOneIntraControlConditionInterruptions = mean(IntraHypothesisOneIntraControlConditionInterruptions)
pstdevIntraHypothesisOneIntraControlConditionInterruptions = pstdev(IntraHypothesisOneIntraControlConditionInterruptions)

differenceBtwH1EXPandH1CONAttend = meanIntraHypothesisOneIntraExperimentationInterruptions - meanIntraHypothesisOneIntraControlConditionInterruptions
fractionalDifferenceBtwH1EXPandH1CONAttend = differenceBtwH1EXPandH1CONAttend/meanIntraHypothesisOneIntraExperimentationInterruptions

InterventionalH1andComparisonH1Attend.insert(0, "Name", "Interruption Lag's Stats BTW Interventional H1 and Comparison H1")
InterventionalH1andComparisonH1Attend.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationInterruptions)
InterventionalH1andComparisonH1Attend.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationInterruptions)
InterventionalH1andComparisonH1Attend.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisOneIntraControlConditionInterruptions)
InterventionalH1andComparisonH1Attend.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisOneIntraControlConditionInterruptions)
InterventionalH1andComparisonH1Attend.insert(13, "Diff In Seconds", differenceBtwH1EXPandH1CONAttend)
InterventionalH1andComparisonH1Attend.insert(14, "Fraction In Seconds", fractionalDifferenceBtwH1EXPandH1CONAttend)

# Accuracy's Stats BTW Interventional H1 and Comparison H1
InterventionalH1andComparisonH1Accuracy = statisticize.ttest(IntraHypothesisOneIntraExperimentalConditionAccuracies,
                                              IntraHypothesisOneIntraControlConditionAccuracies, paired=False, alternative="greater")

meanIntraHypothesisOneIntraExperimentationAccuracies = mean(IntraHypothesisOneIntraExperimentalConditionAccuracies)
pstdevIntraHypothesisOneIntraExperimentationAccuracies = pstdev(IntraHypothesisOneIntraExperimentalConditionAccuracies)

meanIntraHypothesisOneIntraControlConditionAccuracies = mean(IntraHypothesisOneIntraControlConditionAccuracies)
pstdevIntraHypothesisOneIntraControlConditionAccuracies = pstdev(IntraHypothesisOneIntraControlConditionAccuracies)

differenceBtwH1EXPandH1CONAccuracy = meanIntraHypothesisOneIntraExperimentationAccuracies - meanIntraHypothesisOneIntraControlConditionAccuracies
fractionalDifferenceBtwH1EXPandH1CONAccuracy = differenceBtwH1EXPandH1CONAccuracy

InterventionalH1andComparisonH1Accuracy.insert(0, "Name", "Accuracy's Stats BTW Interventional H1 and Comparison H1")
InterventionalH1andComparisonH1Accuracy.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationAccuracies)
InterventionalH1andComparisonH1Accuracy.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationAccuracies)
InterventionalH1andComparisonH1Accuracy.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisOneIntraControlConditionAccuracies)
InterventionalH1andComparisonH1Accuracy.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisOneIntraControlConditionAccuracies)
InterventionalH1andComparisonH1Accuracy.insert(13, "Diff In Seconds", differenceBtwH1EXPandH1CONAccuracy)
InterventionalH1andComparisonH1Accuracy.insert(14, "Fraction In Seconds", fractionalDifferenceBtwH1EXPandH1CONAccuracy)

# Speed's Stats BTW Interventional H1 and Comparison H1
InterventionalH1andComparisonH1Speed = statisticize.ttest(IntraHypothesisOneIntraExperimentalConditionSpeeds,
                                              IntraHypothesisOneIntraControlConditionSpeeds, paired=False, alternative="greater")

meanIntraHypothesisOneIntraExperimentationSpeeds = mean(IntraHypothesisOneIntraExperimentalConditionSpeeds)
pstdevIntraHypothesisOneIntraExperimentationSpeeds = pstdev(IntraHypothesisOneIntraExperimentalConditionSpeeds)

meanIntraHypothesisOneIntraControlConditionSpeeds = mean(IntraHypothesisOneIntraControlConditionSpeeds)
pstdevIntraHypothesisOneIntraControlConditionSpeeds = pstdev(IntraHypothesisOneIntraControlConditionSpeeds)

differenceBtwH1EXPandH1CONSpeed = meanIntraHypothesisOneIntraExperimentationSpeeds - meanIntraHypothesisOneIntraControlConditionSpeeds
fractionalDifferenceBtwH1EXPandH1CONSpeed = differenceBtwH1EXPandH1CONSpeed/meanIntraHypothesisOneIntraExperimentationSpeeds

InterventionalH1andComparisonH1Speed.insert(0, "Name", "Speed's Stats BTW Interventional H1 and Comparison H1")
InterventionalH1andComparisonH1Speed.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationSpeeds)
InterventionalH1andComparisonH1Speed.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationSpeeds)
InterventionalH1andComparisonH1Speed.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisOneIntraControlConditionSpeeds)
InterventionalH1andComparisonH1Speed.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisOneIntraControlConditionSpeeds)
InterventionalH1andComparisonH1Speed.insert(13, "Diff In Seconds", differenceBtwH1EXPandH1CONSpeed)
InterventionalH1andComparisonH1Speed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwH1EXPandH1CONSpeed)



# Resumption Lag's Stats BTW Interventional H2 and Comparison H2
InterventionalH2andComparisonH2Resume = statisticize.ttest(IntraHypothesisTwoIntraExperimentalConditionResumptions,
                                              IntraHypothesisTwoIntraControlConditionResumptions, paired=False, alternative="less")

meanIntraHypothesisTwoIntraExperimentationResumptions = mean(IntraHypothesisTwoIntraExperimentalConditionResumptions)
pstdevIntraHypothesisTwoIntraExperimentationResumptions = pstdev(IntraHypothesisTwoIntraExperimentalConditionResumptions)

meanIntraHypothesisTwoIntraControlConditionResumptions = mean(IntraHypothesisTwoIntraControlConditionResumptions)
pstdevIntraHypothesisTwoIntraControlConditionResumptions = pstdev(IntraHypothesisTwoIntraControlConditionResumptions)

differenceBtwH2EXPandH2CONResume = meanIntraHypothesisTwoIntraExperimentationResumptions - meanIntraHypothesisTwoIntraControlConditionResumptions
fractionalDifferenceBtwH2EXPandH2CONResume = differenceBtwH2EXPandH2CONResume/meanIntraHypothesisTwoIntraExperimentationResumptions

InterventionalH2andComparisonH2Resume.insert(0, "Name", "Resumption Lag's Stats BTW Interventional H2 and Comparison H2")
InterventionalH2andComparisonH2Resume.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisTwoIntraExperimentationResumptions)
InterventionalH2andComparisonH2Resume.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisTwoIntraExperimentationResumptions)
InterventionalH2andComparisonH2Resume.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraControlConditionResumptions)
InterventionalH2andComparisonH2Resume.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraControlConditionResumptions)
InterventionalH2andComparisonH2Resume.insert(13, "Diff In Seconds", differenceBtwH2EXPandH2CONResume)
InterventionalH2andComparisonH2Resume.insert(14, "Fraction In Seconds", fractionalDifferenceBtwH2EXPandH2CONResume)

# Interruption Lag's Stats BTW Interventional H2 and Comparison H2
InterventionalH2andComparisonH2Attend = statisticize.ttest(IntraHypothesisTwoIntraExperimentalConditionInterruptions,
                                              IntraHypothesisTwoIntraControlConditionInterruptions, paired=False, alternative="less")

meanIntraHypothesisTwoIntraExperimentationInterruptions = mean(IntraHypothesisTwoIntraExperimentalConditionInterruptions)
pstdevIntraHypothesisTwoIntraExperimentationInterruptions = pstdev(IntraHypothesisTwoIntraExperimentalConditionInterruptions)

meanIntraHypothesisTwoIntraControlConditionInterruptions = mean(IntraHypothesisTwoIntraControlConditionInterruptions)
pstdevIntraHypothesisTwoIntraControlConditionInterruptions = pstdev(IntraHypothesisTwoIntraControlConditionInterruptions)

differenceBtwH2EXPandH2CONAttend = meanIntraHypothesisTwoIntraExperimentationInterruptions - meanIntraHypothesisTwoIntraControlConditionInterruptions
fractionalDifferenceBtwH2EXPandH2CONAttend = differenceBtwH2EXPandH2CONAttend/meanIntraHypothesisTwoIntraExperimentationInterruptions

InterventionalH2andComparisonH2Attend.insert(0, "Name", "Interruption Lag's Stats BTW Interventional H2 and Comparison H2")
InterventionalH2andComparisonH2Attend.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisTwoIntraExperimentationInterruptions)
InterventionalH2andComparisonH2Attend.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisTwoIntraExperimentationInterruptions)
InterventionalH2andComparisonH2Attend.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraControlConditionInterruptions)
InterventionalH2andComparisonH2Attend.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraControlConditionInterruptions)
InterventionalH2andComparisonH2Attend.insert(13, "Diff In Seconds", differenceBtwH2EXPandH2CONAttend)
InterventionalH2andComparisonH2Attend.insert(14, "Fraction In Seconds", fractionalDifferenceBtwH2EXPandH2CONAttend)

# Accuracy's Stats BTW Interventional H2 and Comparison H2
InterventionalH2andComparisonH2Accuracy = statisticize.ttest(IntraHypothesisTwoIntraExperimentalConditionAccuracies,
                                              IntraHypothesisTwoIntraControlConditionAccuracies, paired=False, alternative="less")

meanIntraHypothesisTwoIntraExperimentationAccuracies = mean(IntraHypothesisTwoIntraExperimentalConditionAccuracies)
pstdevIntraHypothesisTwoIntraExperimentationAccuracies = pstdev(IntraHypothesisTwoIntraExperimentalConditionAccuracies)

meanIntraHypothesisTwoIntraControlConditionAccuracies = mean(IntraHypothesisTwoIntraControlConditionAccuracies)
pstdevIntraHypothesisTwoIntraControlConditionAccuracies = pstdev(IntraHypothesisTwoIntraControlConditionAccuracies)

differenceBtwH2EXPandH2CONAccuracy = meanIntraHypothesisTwoIntraExperimentationAccuracies - meanIntraHypothesisTwoIntraControlConditionAccuracies
fractionalDifferenceBtwH2EXPandH2CONAccuracy = differenceBtwH2EXPandH2CONAccuracy

InterventionalH2andComparisonH2Accuracy.insert(0, "Name", "Accuracies Lag's Stats BTW H2 and H2 Experimental Intervention")
InterventionalH2andComparisonH2Accuracy.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisTwoIntraExperimentationAccuracies)
InterventionalH2andComparisonH2Accuracy.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisTwoIntraExperimentationAccuracies)
InterventionalH2andComparisonH2Accuracy.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraControlConditionAccuracies)
InterventionalH2andComparisonH2Accuracy.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraControlConditionAccuracies)
InterventionalH2andComparisonH2Accuracy.insert(13, "Diff In Seconds", differenceBtwH2EXPandH2CONAccuracy)
InterventionalH2andComparisonH2Accuracy.insert(14, "Fraction In Seconds", fractionalDifferenceBtwH2EXPandH2CONAccuracy)

# Speed's Stats BTW Interventional H2 and Comparison H2
InterventionalH2andComparisonH2Speed = statisticize.ttest(IntraHypothesisTwoIntraExperimentalConditionSpeeds,
                                              IntraHypothesisTwoIntraControlConditionSpeeds, paired=False, alternative="less")

meanIntraHypothesisTwoIntraExperimentationSpeeds = mean(IntraHypothesisTwoIntraExperimentalConditionSpeeds)
pstdevIntraHypothesisTwoIntraExperimentationSpeeds = pstdev(IntraHypothesisTwoIntraExperimentalConditionSpeeds)

meanIntraHypothesisTwoIntraControlConditionSpeeds = mean(IntraHypothesisTwoIntraControlConditionSpeeds)
pstdevIntraHypothesisTwoIntraControlConditionSpeeds = pstdev(IntraHypothesisTwoIntraControlConditionSpeeds)

differenceBtwH2EXPandH2CONSpeed = meanIntraHypothesisTwoIntraExperimentationSpeeds - meanIntraHypothesisTwoIntraControlConditionSpeeds
fractionalDifferenceBtwH2EXPandH2CONSpeed = differenceBtwH2EXPandH2CONSpeed/meanIntraHypothesisTwoIntraExperimentationSpeeds

InterventionalH2andComparisonH2Speed.insert(0, "Name", "Speeds Lag's Stats BTW H2 and H2 Experimental Intervention")
InterventionalH2andComparisonH2Speed.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisTwoIntraExperimentationSpeeds)
InterventionalH2andComparisonH2Speed.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisTwoIntraExperimentationSpeeds)
InterventionalH2andComparisonH2Speed.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisTwoIntraControlConditionSpeeds)
InterventionalH2andComparisonH2Speed.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisTwoIntraControlConditionSpeeds)
InterventionalH2andComparisonH2Speed.insert(13, "Diff In Seconds", differenceBtwH2EXPandH2CONSpeed)
InterventionalH2andComparisonH2Speed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwH2EXPandH2CONSpeed)

space = [""]
MetricsH1ExperimentalControlHeader = pd.DataFrame(data={"": space})
MetricsH1ExperimentalControlHeader.insert(0, "Name", "Stats BTW Interventional H1 and Comparison H1 Across Conditions")

space = [""]
MetricsH2ExperimentalControlHeader = pd.DataFrame(data={"": space})
MetricsH2ExperimentalControlHeader.insert(0, "Name", "Stats BTW Interventional H2 and Comparison H2 Across Conditions")



# All 120 datapoints per metric per participant in the Experimental condition
ExperimentalInterventionResumptionLagsAssessment =\
ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment +\
ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment+\
ExpH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment+\
ExpH2MathStroopMathDrawCollectedSumResumptionLagsAssessment+\
ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment+\
ExpH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment

ExperimentalInterventionResumptionLagsTesting =\
ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting+\
ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting+\
ExpH2StroopMathStroopDrawCollectedSumResumptionLagsTesting+\
ExpH2MathStroopMathDrawCollectedSumResumptionLagsTesting+\
ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting+\
ExpH2MathStroopMathHanoiCollectedSumResumptionLagsTesting

ExperimentalInterventionInterruptionLagsAssessment =\
ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment+\
ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment+\
ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment+\
ExpH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment+\
ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment+\
ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment

ExperimentalInterventionInterruptionLagsTesting =\
ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting+\
ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting+\
ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting+\
ExpH2MathStroopMathDrawCollectedSumInterruptionLagsTesting+\
ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting+\
ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting

ExperimentalInterventionAccuraciesAssessment =\
ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment+\
ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment+\
ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment+\
ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment+\
ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment+\
ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment

ExperimentalInterventionAccuraciesTesting =\
ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting+\
ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting+\
ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting+\
ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting+\
ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting+\
ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting

ExperimentalInterventionSpeedAssessment = \
ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment +\
ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment+\
ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment+\
ExpH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment+\
ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment+\
ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment

ExperimentalInterventionSpeedTesting = \
ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting + \
ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting + \
ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting + \
ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting + \
ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting + \
ExpH2MathStroopMathDrawCollectedSumsCompletionTimesTesting + \
ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting + \
ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting


# All 120 datapoints per metric per participant in the Control condition
ControlComparisonResumptionLagsAssessment =\
ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment +\
ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment+\
ControlH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment+\
ControlH2MathStroopMathDrawCollectedSumResumptionLagsAssessment+\
ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment+\
ControlH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment

ControlComparisonResumptionLagsTesting =\
ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting+\
ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting+\
ControlH2StroopMathStroopDrawCollectedSumResumptionLagsTesting+\
ControlH2MathStroopMathDrawCollectedSumResumptionLagsTesting+\
ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting+\
ControlH2MathStroopMathHanoiCollectedSumResumptionLagsTesting

ControlComparisonInterruptionLagsAssessment =\
ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment+\
ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment+\
ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment+\
ControlH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment+\
ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment+\
ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment

ControlComparisonInterruptionLagsTesting =\
ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting+\
ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting+\
ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting+\
ControlH2MathStroopMathDrawCollectedSumInterruptionLagsTesting+\
ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting+\
ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting

ControlComparisonAccuraciesAssessment =\
ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment+\
ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment+\
ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment+\
ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment+\
ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment+\
ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment

ControlComparisonAccuraciesTesting =\
ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting+\
ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting+\
ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting+\
ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting+\
ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting+\
ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting

ControlComparisonSpeedAssessment = \
ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment +\
ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment+\
ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment+\
ControlH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment+\
ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment+\
ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment

ControlComparisonSpeedTesting = \
ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting + \
ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting + \
ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting + \
ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting + \
ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting + \
ControlH2MathStroopMathDrawCollectedSumsCompletionTimesTesting + \
ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting + \
ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting

space = [""]
AssessTestingInExperimentalHeader = pd.DataFrame(data={"": space})
AssessTestingInExperimentalHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing Within Experimental Intervention")

space = [""]
AssessTestingInControlHeader = pd.DataFrame(data={"": space})
AssessTestingInControlHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing Within Control Comparison")

# space = [""]
# spacer = pd.DataFrame(data={"": space})
# spacer.insert(0, "Name", "Metrics Stats BTW Assessment and Testing Within Experimental Conditions")


# Experimental Intervention Stats
ExperimentalInterventionResumptionLags = statisticize.ttest(ExperimentalInterventionResumptionLagsAssessment,
                                                 ExperimentalInterventionResumptionLagsTesting, paired=True, alternative="greater")

meanExperimentalInterventionResumptionLagsAssessment = mean(ExperimentalInterventionResumptionLagsAssessment)
pstdevExperimentalInterventionResumptionLagsAssessment = pstdev(ExperimentalInterventionResumptionLagsAssessment)

meanExperimentalInterventionResumptionLagsTesting = mean(ExperimentalInterventionResumptionLagsTesting)
pstdevExperimentalInterventionResumptionLagsTesting = pstdev(ExperimentalInterventionResumptionLagsTesting)

differenceBtwAssessResume = meanExperimentalInterventionResumptionLagsAssessment - meanExperimentalInterventionResumptionLagsTesting
fractionalDifferenceBtwAssessResume = differenceBtwAssessResume/meanExperimentalInterventionResumptionLagsAssessment

ExperimentalInterventionResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Within Experimental Intervention")
ExperimentalInterventionResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionResumptionLagsAssessment)
ExperimentalInterventionResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionResumptionLagsAssessment)
ExperimentalInterventionResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionResumptionLagsTesting)
ExperimentalInterventionResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionResumptionLagsTesting)
ExperimentalInterventionResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResume)
ExperimentalInterventionResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResume)


ExperimentalInterventionInterruptionLags = statisticize.ttest(ExperimentalInterventionInterruptionLagsAssessment,
                                                 ExperimentalInterventionInterruptionLagsTesting, paired=True, alternative="greater")

meanExperimentalInterventionInterruptionLagsAssessment = mean(ExperimentalInterventionInterruptionLagsAssessment)
pstdevExperimentalInterventionInterruptionLagsAssessment = pstdev(ExperimentalInterventionInterruptionLagsAssessment)

meanExperimentalInterventionInterruptionLagsTesting = mean(ExperimentalInterventionInterruptionLagsTesting)
pstdevExperimentalInterventionInterruptionLagsTesting = pstdev(ExperimentalInterventionInterruptionLagsTesting)

differenceBtwAssessAttend = meanExperimentalInterventionInterruptionLagsAssessment - meanExperimentalInterventionInterruptionLagsTesting
fractionalDifferenceBtwAssessAttend = differenceBtwAssessAttend/meanExperimentalInterventionInterruptionLagsAssessment

ExperimentalInterventionInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Within Experimental Intervention")
ExperimentalInterventionInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionInterruptionLagsAssessment)
ExperimentalInterventionInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionInterruptionLagsAssessment)
ExperimentalInterventionInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionInterruptionLagsTesting)
ExperimentalInterventionInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionInterruptionLagsTesting)
ExperimentalInterventionInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttend)
ExperimentalInterventionInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttend)


ExperimentalInterventionAccuracies = statisticize.ttest(ExperimentalInterventionAccuraciesAssessment,
                                                 ExperimentalInterventionAccuraciesTesting, paired=True, alternative="greater")

meanExperimentalInterventionAccuraciesAssessment = mean(ExperimentalInterventionAccuraciesAssessment)
pstdevExperimentalInterventionAccuraciesAssessment = pstdev(ExperimentalInterventionAccuraciesAssessment)

meanExperimentalInterventionAccuraciesTesting = mean(ExperimentalInterventionAccuraciesTesting)
pstdevExperimentalInterventionAccuraciesTesting = pstdev(ExperimentalInterventionAccuraciesTesting)

differenceBtwAssessAccuracies = meanExperimentalInterventionAccuraciesAssessment - meanExperimentalInterventionAccuraciesTesting
fractionalDifferenceBtwAssessAccuracies = differenceBtwAssessAccuracies

ExperimentalInterventionAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Within Experimental Intervention")
ExperimentalInterventionAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionAccuraciesAssessment)
ExperimentalInterventionAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionAccuraciesAssessment)
ExperimentalInterventionAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionAccuraciesTesting)
ExperimentalInterventionAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionAccuraciesTesting)
ExperimentalInterventionAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuracies)
ExperimentalInterventionAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuracies)

ExperimentalInterventionSpeed = statisticize.ttest(ExperimentalInterventionSpeedAssessment,
                                                 ExperimentalInterventionSpeedTesting, paired=True, alternative="greater")

meanExperimentalInterventionSpeedAssessment = mean(ExperimentalInterventionSpeedAssessment)
pstdevExperimentalInterventionSpeedAssessment = pstdev(ExperimentalInterventionSpeedAssessment)

meanExperimentalInterventionSpeedTesting = mean(ExperimentalInterventionSpeedTesting)
pstdevExperimentalInterventionSpeedTesting = pstdev(ExperimentalInterventionSpeedTesting)

differenceBtwAssessSpeeds = meanExperimentalInterventionSpeedAssessment - meanExperimentalInterventionSpeedTesting
fractionalDifferenceBtwAssessSpeeds = differenceBtwAssessSpeeds/meanExperimentalInterventionSpeedAssessment

ExperimentalInterventionSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Within Experimental Intervention")
ExperimentalInterventionSpeed.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionSpeedAssessment)
ExperimentalInterventionSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionSpeedAssessment)
ExperimentalInterventionSpeed.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionSpeedTesting)
ExperimentalInterventionSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionSpeedTesting)
ExperimentalInterventionSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeeds)
ExperimentalInterventionSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeeds)



# Control Comparison Stats
ControlComparisonResumptionLags = statisticize.ttest(ControlComparisonResumptionLagsAssessment,
                                                 ControlComparisonResumptionLagsTesting, paired=True, alternative="greater")

meanControlComparisonResumptionLagsAssessment = mean(ControlComparisonResumptionLagsAssessment)
pstdevControlComparisonResumptionLagsAssessment = pstdev(ControlComparisonResumptionLagsAssessment)

meanControlComparisonResumptionLagsTesting = mean(ControlComparisonResumptionLagsTesting)
pstdevControlComparisonResumptionLagsTesting = pstdev(ControlComparisonResumptionLagsTesting)

differenceBtwAssessResumeControl = meanControlComparisonResumptionLagsAssessment - meanControlComparisonResumptionLagsTesting
fractionalDifferenceBtwAssessResumeControl = differenceBtwAssessResumeControl/meanControlComparisonResumptionLagsAssessment

ControlComparisonResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Within Control Comparison")
ControlComparisonResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonResumptionLagsAssessment)
ControlComparisonResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonResumptionLagsAssessment)
ControlComparisonResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonResumptionLagsTesting)
ControlComparisonResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonResumptionLagsTesting)
ControlComparisonResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResumeControl)
ControlComparisonResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeControl)


ControlComparisonInterruptionLags = statisticize.ttest(ControlComparisonInterruptionLagsAssessment,
                                                 ControlComparisonInterruptionLagsTesting, paired=True, alternative="greater")

meanControlComparisonInterruptionLagsAssessment = mean(ControlComparisonInterruptionLagsAssessment)
pstdevControlComparisonInterruptionLagsAssessment = pstdev(ControlComparisonInterruptionLagsAssessment)

meanControlComparisonInterruptionLagsTesting = mean(ControlComparisonInterruptionLagsTesting)
pstdevControlComparisonInterruptionLagsTesting = pstdev(ControlComparisonInterruptionLagsTesting)

differenceBtwAssessAttendControl = meanControlComparisonInterruptionLagsAssessment - meanControlComparisonInterruptionLagsTesting
fractionalDifferenceBtwAssessAttendControl = differenceBtwAssessAttendControl/meanControlComparisonInterruptionLagsAssessment

ControlComparisonInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Within Experimental Intervention")
ControlComparisonInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonInterruptionLagsAssessment)
ControlComparisonInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonInterruptionLagsAssessment)
ControlComparisonInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonInterruptionLagsTesting)
ControlComparisonInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonInterruptionLagsTesting)
ControlComparisonInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttendControl)
ControlComparisonInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendControl)


ControlComparisonAccuracies = statisticize.ttest(ControlComparisonAccuraciesAssessment,
                                                 ControlComparisonAccuraciesTesting, paired=True, alternative="greater")

meanControlComparisonAccuraciesAssessment = mean(ControlComparisonAccuraciesAssessment)
pstdevControlComparisonAccuraciesAssessment = pstdev(ControlComparisonAccuraciesAssessment)

meanControlComparisonAccuraciesTesting = mean(ControlComparisonAccuraciesTesting)
pstdevControlComparisonAccuraciesTesting = pstdev(ControlComparisonAccuraciesTesting)

differenceBtwAssessAccuraciesControl = meanControlComparisonAccuraciesAssessment - meanControlComparisonAccuraciesTesting
fractionalDifferenceBtwAssessAccuraciesControl = differenceBtwAssessAccuraciesControl

ControlComparisonAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Within Control Comparison")
ControlComparisonAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonAccuraciesAssessment)
ControlComparisonAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonAccuraciesAssessment)
ControlComparisonAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonAccuraciesTesting)
ControlComparisonAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonAccuraciesTesting)
ControlComparisonAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesControl)
ControlComparisonAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesControl)


ControlComparisonSpeed = statisticize.ttest(ControlComparisonSpeedAssessment,
                                                 ControlComparisonSpeedTesting, paired=True, alternative="greater")

meanControlComparisonSpeedAssessment = mean(ControlComparisonSpeedAssessment)
pstdevControlComparisonSpeedAssessment = pstdev(ControlComparisonSpeedAssessment)

meanControlComparisonSpeedTesting = mean(ControlComparisonSpeedTesting)
pstdevControlComparisonSpeedTesting = pstdev(ControlComparisonSpeedTesting)

differenceBtwAssessSpeedsControl = meanControlComparisonSpeedAssessment - meanControlComparisonSpeedTesting
fractionalDifferenceBtwAssessSpeedsControl = differenceBtwAssessSpeedsControl/meanControlComparisonSpeedAssessment

ControlComparisonSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Within Experimental Intervention")
ControlComparisonSpeed.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonSpeedAssessment)
ControlComparisonSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonSpeedAssessment)
ControlComparisonSpeed.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonSpeedTesting)
ControlComparisonSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonSpeedTesting)
ControlComparisonSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsControl)
ControlComparisonSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsControl)

# Stats Sorted and Grouped by Hypothesis
# All 60 datapoints per metric per participant per hypothesis within the Experimental Intervention
# Top left quadrant resumptions Assessement
ExperimentalInterventionResumptionLagsAssessmentH1 =\
ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment +\
ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment
# Bottom left quadrant resumptions Assessement
ExperimentalInterventionResumptionLagsAssessmentH2 =\
ExpH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment+\
ExpH2MathStroopMathDrawCollectedSumResumptionLagsAssessment+\
ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment+\
ExpH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment


# Top left quadrant resumptions Testing
ExperimentalInterventionResumptionLagsTestingH1 =\
ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting+\
ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting
# Bottom left quadrant resumptions Testing
ExperimentalInterventionResumptionLagsTestingH2 =\
ExpH2StroopMathStroopDrawCollectedSumResumptionLagsTesting+\
ExpH2MathStroopMathDrawCollectedSumResumptionLagsTesting+\
ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting+\
ExpH2MathStroopMathHanoiCollectedSumResumptionLagsTesting


# Top left quadrant Interruption Assessement
ExperimentalInterventionInterruptionLagsAssessmentH1 =\
ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment+\
ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment
# Bottom left quadrant Interruption Assessement
ExperimentalInterventionInterruptionLagsAssessmentH2 =\
ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment+\
ExpH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment+\
ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment+\
ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment


# Top left quadrant Interruption Testing
ExperimentalInterventionInterruptionLagsTestingH1 =\
ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting+\
ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting
# Bottom left quadrant Interruption Testing
ExperimentalInterventionInterruptionLagsTestingH2 =\
ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting+\
ExpH2MathStroopMathDrawCollectedSumInterruptionLagsTesting+\
ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting+\
ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting


ExperimentalInterventionAccuraciesAssessmentH1 =\
ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment+\
ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment
ExperimentalInterventionAccuraciesAssessmentH2 =\
ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment+\
ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment+\
ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment+\
ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment


ExperimentalInterventionAccuraciesTestingH1 =\
ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting+\
ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting
ExperimentalInterventionAccuraciesTestingH2 =\
ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting+\
ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting+\
ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting+\
ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting


ExperimentalInterventionSpeedAssessmentH1 = \
ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment +\
ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment
ExperimentalInterventionSpeedAssessmentH2 = \
ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment+\
ExpH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment+\
ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment+\
ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment


ExperimentalInterventionSpeedTestingH1 = \
ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting + \
ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting + \
ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting + \
ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting
ExperimentalInterventionSpeedTestingH2 = \
ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting + \
ExpH2MathStroopMathDrawCollectedSumsCompletionTimesTesting + \
ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting + \
ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting



# All 60 datapoints per metric per participant per hypothesis within the Control Comparison
ControlComparisonResumptionLagsAssessmentH1 =\
ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment +\
ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment
ControlComparisonResumptionLagsAssessmentH2 =\
ControlH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment+\
ControlH2MathStroopMathDrawCollectedSumResumptionLagsAssessment+\
ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment+\
ControlH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment

ControlComparisonResumptionLagsTestingH1 =\
ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting+\
ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting
ControlComparisonResumptionLagsTestingH2 =\
ControlH2StroopMathStroopDrawCollectedSumResumptionLagsTesting+\
ControlH2MathStroopMathDrawCollectedSumResumptionLagsTesting+\
ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting+\
ControlH2MathStroopMathHanoiCollectedSumResumptionLagsTesting

ControlComparisonInterruptionLagsAssessmentH1 =\
ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment+\
ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment
ControlComparisonInterruptionLagsAssessmentH2 =\
ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment+\
ControlH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment+\
ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment+\
ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment

ControlComparisonInterruptionLagsTestingH1 =\
ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting+\
ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting
ControlComparisonInterruptionLagsTestingH2 =\
ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting+\
ControlH2MathStroopMathDrawCollectedSumInterruptionLagsTesting+\
ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting+\
ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting

ControlComparisonAccuraciesAssessmentH1 =\
ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment+\
ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment
ControlComparisonAccuraciesAssessmentH2 =\
ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment+\
ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment+\
ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment+\
ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment

ControlComparisonAccuraciesTestingH1 =\
ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting+\
ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting
ControlComparisonAccuraciesTestingH2 =\
ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting+\
ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting+\
ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting+\
ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting

ControlComparisonSpeedAssessmentH1 = \
ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment +\
ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment
ControlComparisonSpeedAssessmentH2 = \
ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment+\
ControlH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment+\
ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment+\
ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment

ControlComparisonSpeedTestingH1 = \
ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting + \
ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting + \
ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting + \
ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting
ControlComparisonSpeedTestingH2 = \
ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting + \
ControlH2MathStroopMathDrawCollectedSumsCompletionTimesTesting + \
ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting + \
ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting


space = [""]
AssessTestingInH1ExperimentalHeader = pd.DataFrame(data={"": space})
AssessTestingInH1ExperimentalHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing Within H1 of Experimental Intervention")

space = [""]
AssessTestingInH2ExperimentalHeader = pd.DataFrame(data={"": space})
AssessTestingInH2ExperimentalHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing Within H2 of Experimental Intervention")

space = [""]
AssessTestingInH1ControlHeader = pd.DataFrame(data={"": space})
AssessTestingInH1ControlHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing Within H1 of Control Comparison")

space = [""]
AssessTestingInH2ControlHeader = pd.DataFrame(data={"": space})
AssessTestingInH2ControlHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing Within H2 of Control Comparison")


# H1 of Experimental Intervention Resumption Stats
H1ExperimentalInterventionResumptionLags = statisticize.ttest(ExperimentalInterventionResumptionLagsAssessmentH1,
                                                 ExperimentalInterventionResumptionLagsTestingH1, paired=True, alternative="greater")

meanExperimentalInterventionResumptionLagsAssessmentH1 = mean(ExperimentalInterventionResumptionLagsAssessmentH1)
pstdevExperimentalInterventionResumptionLagsAssessmentH1 = pstdev(ExperimentalInterventionResumptionLagsAssessmentH1)

meanExperimentalInterventionResumptionLagsTestingH1 = mean(ExperimentalInterventionResumptionLagsTestingH1)
pstdevExperimentalInterventionResumptionLagsTestingH1 = pstdev(ExperimentalInterventionResumptionLagsTestingH1)

differenceBtwAssessResumeH1 = meanExperimentalInterventionResumptionLagsAssessmentH1 - meanExperimentalInterventionResumptionLagsTestingH1
fractionalDifferenceBtwAssessResumeH1 = differenceBtwAssessResumeH1/meanExperimentalInterventionResumptionLagsAssessmentH1

H1ExperimentalInterventionResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Within H1 of Experimental Intervention")
H1ExperimentalInterventionResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionResumptionLagsAssessmentH1)
H1ExperimentalInterventionResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionResumptionLagsAssessmentH1)
H1ExperimentalInterventionResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionResumptionLagsTestingH1)
H1ExperimentalInterventionResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionResumptionLagsTestingH1)
H1ExperimentalInterventionResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResumeH1)
H1ExperimentalInterventionResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeH1)


# H2 of Experimental Intervention Resumption Stats
H2ExperimentalInterventionResumptionLags = statisticize.ttest(ExperimentalInterventionResumptionLagsAssessmentH2,
                                                 ExperimentalInterventionResumptionLagsTestingH2, paired=True, alternative="greater")

meanExperimentalInterventionResumptionLagsAssessmentH2 = mean(ExperimentalInterventionResumptionLagsAssessmentH2)
pstdevExperimentalInterventionResumptionLagsAssessmentH2 = pstdev(ExperimentalInterventionResumptionLagsAssessmentH2)

meanExperimentalInterventionResumptionLagsTestingH2 = mean(ExperimentalInterventionResumptionLagsTestingH2)
pstdevExperimentalInterventionResumptionLagsTestingH2 = pstdev(ExperimentalInterventionResumptionLagsTestingH2)

differenceBtwAssessResumeH2 = meanExperimentalInterventionResumptionLagsAssessmentH2 - meanExperimentalInterventionResumptionLagsTestingH2
fractionalDifferenceBtwAssessResumeH2 = differenceBtwAssessResumeH2/meanExperimentalInterventionResumptionLagsAssessmentH2

H2ExperimentalInterventionResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Within H2 of Experimental Intervention")
H2ExperimentalInterventionResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionResumptionLagsAssessmentH2)
H2ExperimentalInterventionResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionResumptionLagsAssessmentH2)
H2ExperimentalInterventionResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionResumptionLagsTestingH2)
H2ExperimentalInterventionResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionResumptionLagsTestingH2)
H2ExperimentalInterventionResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResumeH2)
H2ExperimentalInterventionResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeH2)


# H1 of Experimental Intervention Interruption Stats
H1ExperimentalInterventionInterruptionLags = statisticize.ttest(ExperimentalInterventionInterruptionLagsAssessmentH1,
                                                 ExperimentalInterventionInterruptionLagsTestingH1, paired=True, alternative="greater")

meanExperimentalInterventionInterruptionLagsAssessmentH1 = mean(ExperimentalInterventionInterruptionLagsAssessmentH1)
pstdevExperimentalInterventionInterruptionLagsAssessmentH1 = pstdev(ExperimentalInterventionInterruptionLagsAssessmentH1)

meanExperimentalInterventionInterruptionLagsTestingH1 = mean(ExperimentalInterventionInterruptionLagsTestingH1)
pstdevExperimentalInterventionInterruptionLagsTestingH1 = pstdev(ExperimentalInterventionInterruptionLagsTestingH1)

differenceBtwAssessAttendH1 = meanExperimentalInterventionInterruptionLagsAssessmentH1 - meanExperimentalInterventionInterruptionLagsTestingH1
fractionalDifferenceBtwAssessAttendH1 = differenceBtwAssessAttendH1/meanExperimentalInterventionInterruptionLagsAssessmentH1

H1ExperimentalInterventionInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Within H1 of Experimental Intervention")
H1ExperimentalInterventionInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionInterruptionLagsAssessmentH1)
H1ExperimentalInterventionInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionInterruptionLagsAssessmentH1)
H1ExperimentalInterventionInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionInterruptionLagsTestingH1)
H1ExperimentalInterventionInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionInterruptionLagsTestingH1)
H1ExperimentalInterventionInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttendH1)
H1ExperimentalInterventionInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendH1)


# H2 of Experimental Intervention Interruption Stats
H2ExperimentalInterventionInterruptionLags = statisticize.ttest(ExperimentalInterventionInterruptionLagsAssessmentH2,
                                                 ExperimentalInterventionInterruptionLagsTestingH2, paired=True, alternative="greater")

meanExperimentalInterventionInterruptionLagsAssessmentH2 = mean(ExperimentalInterventionInterruptionLagsAssessmentH2)
pstdevExperimentalInterventionInterruptionLagsAssessmentH2 = pstdev(ExperimentalInterventionInterruptionLagsAssessmentH2)

meanExperimentalInterventionInterruptionLagsTestingH2 = mean(ExperimentalInterventionInterruptionLagsTestingH2)
pstdevExperimentalInterventionInterruptionLagsTestingH2 = pstdev(ExperimentalInterventionInterruptionLagsTestingH2)

differenceBtwAssessAttendH2 = meanExperimentalInterventionInterruptionLagsAssessmentH2 - meanExperimentalInterventionInterruptionLagsTestingH2
fractionalDifferenceBtwAssessAttendH2 = differenceBtwAssessAttendH2/meanExperimentalInterventionInterruptionLagsAssessmentH2

H2ExperimentalInterventionInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Within H2 of Experimental Intervention")
H2ExperimentalInterventionInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionInterruptionLagsAssessmentH2)
H2ExperimentalInterventionInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionInterruptionLagsAssessmentH2)
H2ExperimentalInterventionInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionInterruptionLagsTestingH2)
H2ExperimentalInterventionInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionInterruptionLagsTestingH2)
H2ExperimentalInterventionInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttendH2)
H2ExperimentalInterventionInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendH2)


# H1 of Experimental Intervention Accuracy Stats
H1ExperimentalInterventionAccuracies = statisticize.ttest(ExperimentalInterventionAccuraciesAssessmentH1,
                                                 ExperimentalInterventionAccuraciesTestingH1, paired=True, alternative="greater")

meanExperimentalInterventionAccuraciesAssessmentH1 = mean(ExperimentalInterventionAccuraciesAssessmentH1)
pstdevExperimentalInterventionAccuraciesAssessmentH1 = pstdev(ExperimentalInterventionAccuraciesAssessmentH1)

meanExperimentalInterventionAccuraciesTestingH1 = mean(ExperimentalInterventionAccuraciesTestingH1)
pstdevExperimentalInterventionAccuraciesTestingH1 = pstdev(ExperimentalInterventionAccuraciesTestingH1)

differenceBtwAssessAccuraciesH1 = meanExperimentalInterventionAccuraciesAssessmentH1 - meanExperimentalInterventionAccuraciesTestingH1
fractionalDifferenceBtwAssessAccuraciesH1 = differenceBtwAssessAccuraciesH1

H1ExperimentalInterventionAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Within H1 of Experimental Intervention")
H1ExperimentalInterventionAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionAccuraciesAssessmentH1)
H1ExperimentalInterventionAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionAccuraciesAssessmentH1)
H1ExperimentalInterventionAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionAccuraciesTestingH1)
H1ExperimentalInterventionAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionAccuraciesTestingH1)
H1ExperimentalInterventionAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesH1)
H1ExperimentalInterventionAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesH1)


# H2 of Experimental Intervention Accuracy Stats
H2ExperimentalInterventionAccuracies = statisticize.ttest(ExperimentalInterventionAccuraciesAssessmentH2,
                                                 ExperimentalInterventionAccuraciesTestingH2, paired=True, alternative="greater")

meanExperimentalInterventionAccuraciesAssessmentH2 = mean(ExperimentalInterventionAccuraciesAssessmentH2)
pstdevExperimentalInterventionAccuraciesAssessmentH2 = pstdev(ExperimentalInterventionAccuraciesAssessmentH2)

meanExperimentalInterventionAccuraciesTestingH2 = mean(ExperimentalInterventionAccuraciesTestingH2)
pstdevExperimentalInterventionAccuraciesTestingH2 = pstdev(ExperimentalInterventionAccuraciesTestingH2)

differenceBtwAssessAccuraciesH2 = meanExperimentalInterventionAccuraciesAssessmentH2 - meanExperimentalInterventionAccuraciesTestingH2
fractionalDifferenceBtwAssessAccuraciesH2 = differenceBtwAssessAccuraciesH2

H2ExperimentalInterventionAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Within H2 of Experimental Intervention")
H2ExperimentalInterventionAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionAccuraciesAssessmentH2)
H2ExperimentalInterventionAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionAccuraciesAssessmentH2)
H2ExperimentalInterventionAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionAccuraciesTestingH2)
H2ExperimentalInterventionAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionAccuraciesTestingH2)
H2ExperimentalInterventionAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesH2)
H2ExperimentalInterventionAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesH2)


# H1 of Experimental Intervention Speed Stats
H1ExperimentalInterventionSpeed = statisticize.ttest(ExperimentalInterventionSpeedAssessmentH1,
                                                 ExperimentalInterventionSpeedTestingH1, paired=True, alternative="greater")

meanExperimentalInterventionSpeedAssessmentH1 = mean(ExperimentalInterventionSpeedAssessmentH1)
pstdevExperimentalInterventionSpeedAssessmentH1 = pstdev(ExperimentalInterventionSpeedAssessmentH1)

meanExperimentalInterventionSpeedTestingH1 = mean(ExperimentalInterventionSpeedTestingH1)
pstdevExperimentalInterventionSpeedTestingH1 = pstdev(ExperimentalInterventionSpeedTestingH1)

differenceBtwAssessSpeedsH1 = meanExperimentalInterventionSpeedAssessmentH1 - meanExperimentalInterventionSpeedTestingH1
fractionalDifferenceBtwAssessSpeedsH1 = differenceBtwAssessSpeedsH1/meanExperimentalInterventionSpeedAssessmentH1

H1ExperimentalInterventionSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Within H1 of Experimental Intervention")
H1ExperimentalInterventionSpeed.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionSpeedAssessmentH1)
H1ExperimentalInterventionSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionSpeedAssessmentH1)
H1ExperimentalInterventionSpeed.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionSpeedTestingH1)
H1ExperimentalInterventionSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionSpeedTestingH1)
H1ExperimentalInterventionSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsH1)
H1ExperimentalInterventionSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsH1)

# H2 of Experimental Intervention Speed Stats
H2ExperimentalInterventionSpeed = statisticize.ttest(ExperimentalInterventionSpeedAssessmentH2,
                                                 ExperimentalInterventionSpeedTestingH2, paired=True, alternative="greater")

meanExperimentalInterventionSpeedAssessmentH2 = mean(ExperimentalInterventionSpeedAssessmentH2)
pstdevExperimentalInterventionSpeedAssessmentH2 = pstdev(ExperimentalInterventionSpeedAssessmentH2)

meanExperimentalInterventionSpeedTestingH2 = mean(ExperimentalInterventionSpeedTestingH2)
pstdevExperimentalInterventionSpeedTestingH2 = pstdev(ExperimentalInterventionSpeedTestingH2)

differenceBtwAssessSpeedsH2 = meanExperimentalInterventionSpeedAssessmentH2 - meanExperimentalInterventionSpeedTestingH2
fractionalDifferenceBtwAssessSpeedsH2 = differenceBtwAssessSpeedsH2/meanExperimentalInterventionSpeedAssessmentH2

H2ExperimentalInterventionSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Within H2 of Experimental Intervention")
H2ExperimentalInterventionSpeed.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionSpeedAssessmentH2)
H2ExperimentalInterventionSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionSpeedAssessmentH2)
H2ExperimentalInterventionSpeed.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionSpeedTestingH2)
H2ExperimentalInterventionSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionSpeedTestingH2)
H2ExperimentalInterventionSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsH2)
H2ExperimentalInterventionSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsH2)



# Control Comparison Stats
# H1 of Control Comparison Resumption Stats
H1ControlComparisonResumptionLags = statisticize.ttest(ControlComparisonResumptionLagsAssessmentH1,
                                                 ControlComparisonResumptionLagsTestingH1, paired=True, alternative="greater")

meanControlComparisonResumptionLagsAssessmentH1 = mean(ControlComparisonResumptionLagsAssessmentH1)
pstdevControlComparisonResumptionLagsAssessmentH1 = pstdev(ControlComparisonResumptionLagsAssessmentH1)

meanControlComparisonResumptionLagsTestingH1 = mean(ControlComparisonResumptionLagsTestingH1)
pstdevControlComparisonResumptionLagsTestingH1 = pstdev(ControlComparisonResumptionLagsTestingH1)

differenceBtwAssessResumeControlH1 = meanControlComparisonResumptionLagsAssessmentH1 - meanControlComparisonResumptionLagsTestingH1
fractionalDifferenceBtwAssessResumeControlH1 = differenceBtwAssessResumeControlH1/meanControlComparisonResumptionLagsAssessmentH1

H1ControlComparisonResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Within H1 Control Comparison")
H1ControlComparisonResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonResumptionLagsAssessmentH1)
H1ControlComparisonResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonResumptionLagsAssessmentH1)
H1ControlComparisonResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonResumptionLagsTestingH1)
H1ControlComparisonResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonResumptionLagsTestingH1)
H1ControlComparisonResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResumeControlH1)
H1ControlComparisonResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeControlH1)


# H2 of Control Comparison Resumption Stats
H2ControlComparisonResumptionLags = statisticize.ttest(ControlComparisonResumptionLagsAssessmentH2,
                                                 ControlComparisonResumptionLagsTestingH2, paired=True, alternative="greater")

meanControlComparisonResumptionLagsAssessmentH2 = mean(ControlComparisonResumptionLagsAssessmentH2)
pstdevControlComparisonResumptionLagsAssessmentH2 = pstdev(ControlComparisonResumptionLagsAssessmentH2)

meanControlComparisonResumptionLagsTestingH2 = mean(ControlComparisonResumptionLagsTestingH2)
pstdevControlComparisonResumptionLagsTestingH2 = pstdev(ControlComparisonResumptionLagsTestingH2)

differenceBtwAssessResumeControlH2 = meanControlComparisonResumptionLagsAssessmentH2 - meanControlComparisonResumptionLagsTestingH2
fractionalDifferenceBtwAssessResumeControlH2 = differenceBtwAssessResumeControlH2/meanControlComparisonResumptionLagsAssessmentH2

H2ControlComparisonResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Within H2 Control Comparison")
H2ControlComparisonResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonResumptionLagsAssessmentH2)
H2ControlComparisonResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonResumptionLagsAssessmentH2)
H2ControlComparisonResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonResumptionLagsTestingH2)
H2ControlComparisonResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonResumptionLagsTestingH2)
H2ControlComparisonResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResumeControlH2)
H2ControlComparisonResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeControlH2)

# H1 of Control Comparison Interruption Stats
H1ControlComparisonInterruptionLags = statisticize.ttest(ControlComparisonInterruptionLagsAssessmentH1,
                                                 ControlComparisonInterruptionLagsTestingH1, paired=True, alternative="greater")

meanControlComparisonInterruptionLagsAssessmentH1 = mean(ControlComparisonInterruptionLagsAssessmentH1)
pstdevControlComparisonInterruptionLagsAssessmentH1 = pstdev(ControlComparisonInterruptionLagsAssessmentH1)

meanControlComparisonInterruptionLagsTestingH1 = mean(ControlComparisonInterruptionLagsTestingH1)
pstdevControlComparisonInterruptionLagsTestingH1 = pstdev(ControlComparisonInterruptionLagsTestingH1)

differenceBtwAssessAttendControlH1 = meanControlComparisonInterruptionLagsAssessmentH1 - meanControlComparisonInterruptionLagsTestingH1
fractionalDifferenceBtwAssessAttendControlH1 = differenceBtwAssessAttendControlH1/meanControlComparisonInterruptionLagsAssessmentH1

H1ControlComparisonInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Within H1 Control Comparison")
H1ControlComparisonInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonInterruptionLagsAssessmentH1)
H1ControlComparisonInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonInterruptionLagsAssessmentH1)
H1ControlComparisonInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonInterruptionLagsTestingH1)
H1ControlComparisonInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonInterruptionLagsTestingH1)
H1ControlComparisonInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttendControlH1)
H1ControlComparisonInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendControlH1)


# H2 of Control Comparison Interruption Stats
H2ControlComparisonInterruptionLags = statisticize.ttest(ControlComparisonInterruptionLagsAssessmentH2,
                                                 ControlComparisonInterruptionLagsTestingH2, paired=True, alternative="greater")

meanControlComparisonInterruptionLagsAssessmentH2 = mean(ControlComparisonInterruptionLagsAssessmentH2)
pstdevControlComparisonInterruptionLagsAssessmentH2 = pstdev(ControlComparisonInterruptionLagsAssessmentH2)

meanControlComparisonInterruptionLagsTestingH2 = mean(ControlComparisonInterruptionLagsTestingH2)
pstdevControlComparisonInterruptionLagsTestingH2 = pstdev(ControlComparisonInterruptionLagsTestingH2)

differenceBtwAssessAttendControlH2 = meanControlComparisonInterruptionLagsAssessmentH2 - meanControlComparisonInterruptionLagsTestingH2
fractionalDifferenceBtwAssessAttendControlH2 = differenceBtwAssessAttendControlH2/meanControlComparisonInterruptionLagsAssessmentH2

H2ControlComparisonInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Within H2 Control Comparison")
H2ControlComparisonInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonInterruptionLagsAssessmentH2)
H2ControlComparisonInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonInterruptionLagsAssessmentH2)
H2ControlComparisonInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonInterruptionLagsTestingH2)
H2ControlComparisonInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonInterruptionLagsTestingH2)
H2ControlComparisonInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttendControlH2)
H2ControlComparisonInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendControlH2)


# H1 of Control Comparison Accuracy Stats
H1ControlComparisonAccuracies = statisticize.ttest(ControlComparisonAccuraciesAssessmentH1,
                                                 ControlComparisonAccuraciesTestingH1, paired=True, alternative="greater")

meanControlComparisonAccuraciesAssessmentH1 = mean(ControlComparisonAccuraciesAssessmentH1)
pstdevControlComparisonAccuraciesAssessmentH1 = pstdev(ControlComparisonAccuraciesAssessmentH1)

meanControlComparisonAccuraciesTestingH1 = mean(ControlComparisonAccuraciesTestingH1)
pstdevControlComparisonAccuraciesTestingH1 = pstdev(ControlComparisonAccuraciesTestingH1)

differenceBtwAssessAccuraciesControlH1 = meanControlComparisonAccuraciesAssessmentH1 - meanControlComparisonAccuraciesTestingH1
fractionalDifferenceBtwAssessAccuraciesControlH1 = differenceBtwAssessAccuraciesControlH1

H1ControlComparisonAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Within H1 Control Comparison")
H1ControlComparisonAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonAccuraciesAssessmentH1)
H1ControlComparisonAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonAccuraciesAssessmentH1)
H1ControlComparisonAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonAccuraciesTestingH1)
H1ControlComparisonAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonAccuraciesTestingH1)
H1ControlComparisonAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesControlH1)
H1ControlComparisonAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesControlH1)


# H2 of Control Comparison Accuracy Stats
H2ControlComparisonAccuracies = statisticize.ttest(ControlComparisonAccuraciesAssessmentH2,
                                                 ControlComparisonAccuraciesTestingH2, paired=True, alternative="greater")

meanControlComparisonAccuraciesAssessmentH2 = mean(ControlComparisonAccuraciesAssessmentH2)
pstdevControlComparisonAccuraciesAssessmentH2 = pstdev(ControlComparisonAccuraciesAssessmentH2)

meanControlComparisonAccuraciesTestingH2 = mean(ControlComparisonAccuraciesTestingH2)
pstdevControlComparisonAccuraciesTestingH2 = pstdev(ControlComparisonAccuraciesTestingH2)

differenceBtwAssessAccuraciesControlH2 = meanControlComparisonAccuraciesAssessmentH2 - meanControlComparisonAccuraciesTestingH2
fractionalDifferenceBtwAssessAccuraciesControlH2 = differenceBtwAssessAccuraciesControlH2

H2ControlComparisonAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Within H2 Control Comparison")
H2ControlComparisonAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonAccuraciesAssessmentH2)
H2ControlComparisonAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonAccuraciesAssessmentH2)
H2ControlComparisonAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonAccuraciesTestingH2)
H2ControlComparisonAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonAccuraciesTestingH2)
H2ControlComparisonAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesControlH2)
H2ControlComparisonAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesControlH2)


# H1 of Control Comparison Speed Stats
H1ControlComparisonSpeed = statisticize.ttest(ControlComparisonSpeedAssessmentH1,
                                                 ControlComparisonSpeedTestingH1, paired=True, alternative="greater")

meanControlComparisonSpeedAssessmentH1 = mean(ControlComparisonSpeedAssessmentH1)
pstdevControlComparisonSpeedAssessmentH1 = pstdev(ControlComparisonSpeedAssessmentH1)

meanControlComparisonSpeedTestingH1 = mean(ControlComparisonSpeedTestingH1)
pstdevControlComparisonSpeedTestingH1 = pstdev(ControlComparisonSpeedTestingH1)

differenceBtwAssessSpeedsControlH1 = meanControlComparisonSpeedAssessmentH1 - meanControlComparisonSpeedTestingH1
fractionalDifferenceBtwAssessSpeedsControlH1 = differenceBtwAssessSpeedsControlH1/meanControlComparisonSpeedAssessmentH1

H1ControlComparisonSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Within H1 Control Comparison")
H1ControlComparisonSpeed.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonSpeedAssessmentH1)
H1ControlComparisonSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonSpeedAssessmentH1)
H1ControlComparisonSpeed.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonSpeedTestingH1)
H1ControlComparisonSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonSpeedTestingH1)
H1ControlComparisonSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsControlH1)
H1ControlComparisonSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsControlH1)


# H2 of Control Comparison Speed Stats
H2ControlComparisonSpeed = statisticize.ttest(ControlComparisonSpeedAssessmentH2,
                                                 ControlComparisonSpeedTestingH2, paired=True, alternative="greater")

meanControlComparisonSpeedAssessmentH2 = mean(ControlComparisonSpeedAssessmentH2)
pstdevControlComparisonSpeedAssessmentH2 = pstdev(ControlComparisonSpeedAssessmentH2)

meanControlComparisonSpeedTestingH2 = mean(ControlComparisonSpeedTestingH2)
pstdevControlComparisonSpeedTestingH2 = pstdev(ControlComparisonSpeedTestingH2)

differenceBtwAssessSpeedsControlH2 = meanControlComparisonSpeedAssessmentH2 - meanControlComparisonSpeedTestingH2
fractionalDifferenceBtwAssessSpeedsControlH2 = differenceBtwAssessSpeedsControlH2/meanControlComparisonSpeedAssessmentH2

H2ControlComparisonSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Within H2 Control Comparison")
H2ControlComparisonSpeed.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonSpeedAssessmentH2)
H2ControlComparisonSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonSpeedAssessmentH2)
H2ControlComparisonSpeed.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonSpeedTestingH2)
H2ControlComparisonSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonSpeedTestingH2)
H2ControlComparisonSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsControlH2)
H2ControlComparisonSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsControlH2)



# Stats Sorted and Grouped by Primary Task
# All 60 datapoints per metric per participant per hypothesis within the Experimental Intervention

ExperimentalInterventionResumptionLagsAssessmentPathTracing=\
ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment +\
ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment+\
ExpH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment+\
ExpH2MathStroopMathDrawCollectedSumResumptionLagsAssessment

ExperimentalInterventionResumptionLagsAssessmentToH=\
ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment+\
ExpH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment

ExperimentalInterventionResumptionLagsTestingPathTracing=\
ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting+\
ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting+\
ExpH2StroopMathStroopDrawCollectedSumResumptionLagsTesting+\
ExpH2MathStroopMathDrawCollectedSumResumptionLagsTesting

ExperimentalInterventionResumptionLagsTestingToH=\
ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting+\
ExpH2MathStroopMathHanoiCollectedSumResumptionLagsTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting

ExperimentalInterventionInterruptionLagsAssessmentPathTracing=\
ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment+\
ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment+\
ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment+\
ExpH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment

ExperimentalInterventionInterruptionLagsAssessmentToH=\
ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment+\
ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment

ExperimentalInterventionInterruptionLagsTestingPathTracing=\
ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting+\
ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting+\
ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting+\
ExpH2MathStroopMathDrawCollectedSumInterruptionLagsTesting

ExperimentalInterventionInterruptionLagsTestingToH=\
ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting+\
ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting

ExperimentalInterventionAccuraciesAssessmentPathTracing=\
ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment+\
ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment+\
ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment+\
ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment

ExperimentalInterventionAccuraciesAssessmentToH=\
ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment+\
ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment

ExperimentalInterventionAccuraciesTestingPathTracing=\
ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting+\
ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting+\
ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting+\
ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting

ExperimentalInterventionAccuraciesTestingToH=\
ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting+\
ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting

ExperimentalInterventionSpeedAssessmentPathTracing= \
ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment +\
ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment+\
ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment+\
ExpH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment

ExperimentalInterventionSpeedAssessmentToH= \
ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment+\
ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment+\
ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment

ExperimentalInterventionSpeedTestingPathTracing= \
ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting + \
ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting + \
ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting + \
ExpH2MathStroopMathDrawCollectedSumsCompletionTimesTesting

ExperimentalInterventionSpeedTestingToH= \
ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting + \
ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting+\
ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting + \
ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting


# All 60 datapoints per metric per participant per hypothesis within the Control Comparison
ControlComparisonResumptionLagsAssessmentPathTracing=\
ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment +\
ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment+\
ControlH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment+\
ControlH2MathStroopMathDrawCollectedSumResumptionLagsAssessment

ControlComparisonResumptionLagsAssessmentToH=\
ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment+\
ControlH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment

ControlComparisonResumptionLagsTestingPathTracing=\
ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting+\
ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting+\
ControlH2StroopMathStroopDrawCollectedSumResumptionLagsTesting+\
ControlH2MathStroopMathDrawCollectedSumResumptionLagsTesting

ControlComparisonResumptionLagsTestingToH=\
ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting+\
ControlH2MathStroopMathHanoiCollectedSumResumptionLagsTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting

ControlComparisonInterruptionLagsAssessmentPathTracing=\
ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment+\
ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment+\
ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment+\
ControlH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment

ControlComparisonInterruptionLagsAssessmentToH=\
ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment+\
ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment

ControlComparisonInterruptionLagsTestingPathTracing=\
ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting+\
ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting+\
ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting+\
ControlH2MathStroopMathDrawCollectedSumInterruptionLagsTesting

ControlComparisonInterruptionLagsTestingToH=\
ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting+\
ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting

ControlComparisonAccuraciesAssessmentPathTracing=\
ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment+\
ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment+\
ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment+\
ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment

ControlComparisonAccuraciesAssessmentToH=\
ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment+\
ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment

ControlComparisonAccuraciesTestingPathTracing=\
ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting+\
ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting+\
ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting+\
ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting

ControlComparisonAccuraciesTestingToH=\
ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting+\
ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting

ControlComparisonSpeedAssessmentPathTracing= \
ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment +\
ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment+\
ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment+\
ControlH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment

ControlComparisonSpeedAssessmentToH= \
ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment+\
ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment+\
ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment

ControlComparisonSpeedTestingPathTracing= \
ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting + \
ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting + \
ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting + \
ControlH2MathStroopMathDrawCollectedSumsCompletionTimesTesting

ControlComparisonSpeedTestingToH= \
ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting + \
ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting+\
ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting + \
ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting


space = [""]
AssessTestingPathTracingExperimentalHeader = pd.DataFrame(data={"": space})
AssessTestingPathTracingExperimentalHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for Path-Tracing of Experimental Intervention")

space = [""]
AssessTestingInToHExperimentalHeader = pd.DataFrame(data={"": space})
AssessTestingInToHExperimentalHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for ToH of Experimental Intervention")

space = [""]
AssessTestingPathTracingControlHeader = pd.DataFrame(data={"": space})
AssessTestingPathTracingControlHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for Path-Tracing of Control Comparison")

space = [""]
AssessTestingToHControlHeader = pd.DataFrame(data={"": space})
AssessTestingToHControlHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for ToH of Control Comparison")



# performance Metric Stats Sorted and Grouped by Hypothesis
# All 120 datapoints per metric per participant per H1 across both conditions
# ------------------------------Resumption
RLH1EXPplusCONAssess = ExperimentalInterventionResumptionLagsAssessmentH1+ControlComparisonResumptionLagsAssessmentH1
RLH1EXPplusCONTest = ExperimentalInterventionResumptionLagsTestingH1+ControlComparisonResumptionLagsTestingH1
# RL H1 EXP plus CON
RLH1EXPplusCONStats = statisticize.ttest(RLH1EXPplusCONAssess,
                                                 RLH1EXPplusCONTest, paired=True, alternative="greater")

meanRLH1EXPplusCONAssess = mean(RLH1EXPplusCONAssess)
pstdevRLH1EXPplusCONAssess = pstdev(RLH1EXPplusCONAssess)

meanRLH1EXPplusCONTest = mean(RLH1EXPplusCONTest)
pstdevRLH1EXPplusCONTest = pstdev(RLH1EXPplusCONTest)

differenceRLH1EXPplusCONTest = meanRLH1EXPplusCONAssess - meanRLH1EXPplusCONTest
fractionalDifferenceRLH1EXPplusCONTest = differenceRLH1EXPplusCONTest/meanRLH1EXPplusCONAssess

RLH1EXPplusCONStats.insert(0, "Name", "Resumption Lag across both Conditions (H1)")
RLH1EXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanRLH1EXPplusCONAssess)
RLH1EXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevRLH1EXPplusCONAssess)
RLH1EXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanRLH1EXPplusCONTest)
RLH1EXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevRLH1EXPplusCONTest)
RLH1EXPplusCONStats.insert(13, "Diff In Seconds", differenceRLH1EXPplusCONTest)
RLH1EXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceRLH1EXPplusCONTest)

# ------------------------------Interruption
ILH1EXPplusCONAssess = ExperimentalInterventionInterruptionLagsAssessmentH1+ControlComparisonInterruptionLagsAssessmentH1
ILH1EXPplusCONTest = ExperimentalInterventionInterruptionLagsTestingH1+ControlComparisonInterruptionLagsTestingH1
# IL H1 EXP plus CON
ILH1EXPplusCONStats = statisticize.ttest(ILH1EXPplusCONAssess,
                                                 ILH1EXPplusCONTest, paired=True, alternative="greater")

meanILH1EXPplusCONAssess = mean(ILH1EXPplusCONAssess)
pstdevILH1EXPplusCONAssess = pstdev(ILH1EXPplusCONAssess)

meanILH1EXPplusCONTest = mean(ILH1EXPplusCONTest)
pstdevILH1EXPplusCONTest = pstdev(ILH1EXPplusCONTest)

differenceILH1EXPplusCONTest = meanILH1EXPplusCONAssess - meanILH1EXPplusCONTest
fractionalDifferenceILH1EXPplusCONTest = differenceILH1EXPplusCONTest/meanILH1EXPplusCONAssess

ILH1EXPplusCONStats.insert(0, "Name", "Interruption Lag across both Conditions (H1)")
ILH1EXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanILH1EXPplusCONAssess)
ILH1EXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevILH1EXPplusCONAssess)
ILH1EXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanILH1EXPplusCONTest)
ILH1EXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevILH1EXPplusCONTest)
ILH1EXPplusCONStats.insert(13, "Diff In Seconds", differenceILH1EXPplusCONTest)
ILH1EXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceILH1EXPplusCONTest)
# ------------------------------Accuracy
ACH1EXPplusCONAssess = ExperimentalInterventionAccuraciesAssessmentH1+ControlComparisonAccuraciesAssessmentH1
ACH1EXPplusCONTest = ExperimentalInterventionAccuraciesTestingH1+ControlComparisonAccuraciesTestingH1
# AC H1 EXP plus CON
ACH1EXPplusCONStats = statisticize.ttest(ACH1EXPplusCONAssess,
                                                 ACH1EXPplusCONTest, paired=True, alternative="greater")

meanACH1EXPplusCONAssess = mean(ACH1EXPplusCONAssess)
pstdevACH1EXPplusCONAssess = pstdev(ACH1EXPplusCONAssess)

meanACH1EXPplusCONTest = mean(ACH1EXPplusCONTest)
pstdevACH1EXPplusCONTest = pstdev(ACH1EXPplusCONTest)

differenceACH1EXPplusCONTest = meanACH1EXPplusCONAssess - meanACH1EXPplusCONTest
fractionalDifferenceACH1EXPplusCONTest = differenceACH1EXPplusCONTest

ACH1EXPplusCONStats.insert(0, "Name", "Accuracy across both Conditions (H1)")
ACH1EXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanACH1EXPplusCONAssess)
ACH1EXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevACH1EXPplusCONAssess)
ACH1EXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanACH1EXPplusCONTest)
ACH1EXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevACH1EXPplusCONTest)
ACH1EXPplusCONStats.insert(13, "Diff In Seconds", differenceACH1EXPplusCONTest)
ACH1EXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceACH1EXPplusCONTest)
# ------------------------------Speed
SPH1EXPplusCONAssess = ExperimentalInterventionSpeedAssessmentH1+ControlComparisonSpeedAssessmentH1
SPH1EXPplusCONTest = ExperimentalInterventionSpeedTestingH1+ControlComparisonSpeedTestingH1
# SP H1 EXP plus CON
SPH1EXPplusCONStats = statisticize.ttest(SPH1EXPplusCONAssess,
                                                 SPH1EXPplusCONTest, paired=True, alternative="greater")

meanSPH1EXPplusCONAssess = mean(SPH1EXPplusCONAssess)
pstdevSPH1EXPplusCONAssess = pstdev(SPH1EXPplusCONAssess)

meanSPH1EXPplusCONTest = mean(SPH1EXPplusCONTest)
pstdevSPH1EXPplusCONTest = pstdev(SPH1EXPplusCONTest)

differenceSPH1EXPplusCONTest = meanSPH1EXPplusCONAssess - meanSPH1EXPplusCONTest
fractionalDifferenceSPH1EXPplusCONTest = differenceSPH1EXPplusCONTest/meanSPH1EXPplusCONAssess

SPH1EXPplusCONStats.insert(0, "Name", "Speed across both Conditions (H1)")
SPH1EXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanSPH1EXPplusCONAssess)
SPH1EXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevSPH1EXPplusCONAssess)
SPH1EXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanSPH1EXPplusCONTest)
SPH1EXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevSPH1EXPplusCONTest)
SPH1EXPplusCONStats.insert(13, "Diff In Seconds", differenceSPH1EXPplusCONTest)
SPH1EXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceSPH1EXPplusCONTest)

space = [""]
PerformanceMetricH1EXPplusCONHeader = pd.DataFrame(data={"": space})
PerformanceMetricH1EXPplusCONHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing Within H1 across both Conditions")


# All 120 datapoints per metric per participant per H2 across both conditions
# ------------------------------Resumption
RLH2EXPplusCONAssess = ExperimentalInterventionResumptionLagsAssessmentH2+ControlComparisonResumptionLagsAssessmentH2
RLH2EXPplusCONTest = ExperimentalInterventionResumptionLagsTestingH2+ControlComparisonResumptionLagsTestingH2
# RL H2 EXP plus CON
RLH2EXPplusCONStats = statisticize.ttest(RLH2EXPplusCONAssess,
                                                 RLH2EXPplusCONTest, paired=True, alternative="greater")

meanRLH2EXPplusCONAssess = mean(RLH2EXPplusCONAssess)
pstdevRLH2EXPplusCONAssess = pstdev(RLH2EXPplusCONAssess)

meanRLH2EXPplusCONTest = mean(RLH2EXPplusCONTest)
pstdevRLH2EXPplusCONTest = pstdev(RLH2EXPplusCONTest)

differenceRLH2EXPplusCONTest = meanRLH2EXPplusCONAssess - meanRLH2EXPplusCONTest
fractionalDifferenceRLH2EXPplusCONTest = differenceRLH2EXPplusCONTest/meanRLH2EXPplusCONAssess

RLH2EXPplusCONStats.insert(0, "Name", "Resumption Lag across both Conditions (H2)")
RLH2EXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanRLH2EXPplusCONAssess)
RLH2EXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevRLH2EXPplusCONAssess)
RLH2EXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanRLH2EXPplusCONTest)
RLH2EXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevRLH2EXPplusCONTest)
RLH2EXPplusCONStats.insert(13, "Diff In Seconds", differenceRLH2EXPplusCONTest)
RLH2EXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceRLH2EXPplusCONTest)
# ------------------------------Interruption
ILH2EXPplusCONAssess = ExperimentalInterventionInterruptionLagsAssessmentH2+ControlComparisonInterruptionLagsAssessmentH2
ILH2EXPplusCONTest = ExperimentalInterventionInterruptionLagsTestingH2+ControlComparisonInterruptionLagsTestingH2
# IL H2 EXP plus CON
ILH2EXPplusCONStats = statisticize.ttest(ILH2EXPplusCONAssess,
                                                 ILH2EXPplusCONTest, paired=True, alternative="greater")

meanILH2EXPplusCONAssess = mean(ILH2EXPplusCONAssess)
pstdevILH2EXPplusCONAssess = pstdev(ILH2EXPplusCONAssess)

meanILH2EXPplusCONTest = mean(ILH2EXPplusCONTest)
pstdevILH2EXPplusCONTest = pstdev(ILH2EXPplusCONTest)

differenceILH2EXPplusCONTest = meanILH2EXPplusCONAssess - meanILH2EXPplusCONTest
fractionalDifferenceILH2EXPplusCONTest = differenceILH2EXPplusCONTest/meanILH2EXPplusCONAssess

ILH2EXPplusCONStats.insert(0, "Name", "Interruption Lag across both Conditions (H2)")
ILH2EXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanILH2EXPplusCONAssess)
ILH2EXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevILH2EXPplusCONAssess)
ILH2EXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanILH2EXPplusCONTest)
ILH2EXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevILH2EXPplusCONTest)
ILH2EXPplusCONStats.insert(13, "Diff In Seconds", differenceILH2EXPplusCONTest)
ILH2EXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceILH2EXPplusCONTest)
# ------------------------------Accuracy
ACH2EXPplusCONAssess = ExperimentalInterventionAccuraciesAssessmentH2+ControlComparisonAccuraciesAssessmentH2
ACH2EXPplusCONTest = ExperimentalInterventionAccuraciesTestingH2+ControlComparisonAccuraciesTestingH2
# AC H2 EXP plus CON
ACH2EXPplusCONStats = statisticize.ttest(ACH2EXPplusCONAssess,
                                                 ACH2EXPplusCONTest, paired=True, alternative="greater")

meanACH2EXPplusCONAssess = mean(ACH2EXPplusCONAssess)
pstdevACH2EXPplusCONAssess = pstdev(ACH2EXPplusCONAssess)

meanACH2EXPplusCONTest = mean(ACH2EXPplusCONTest)
pstdevACH2EXPplusCONTest = pstdev(ACH2EXPplusCONTest)

differenceACH2EXPplusCONTest = meanACH2EXPplusCONAssess - meanACH2EXPplusCONTest
fractionalDifferenceACH2EXPplusCONTest = differenceACH2EXPplusCONTest

ACH2EXPplusCONStats.insert(0, "Name", "Accuracy across both Conditions (H2)")
ACH2EXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanACH2EXPplusCONAssess)
ACH2EXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevACH2EXPplusCONAssess)
ACH2EXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanACH2EXPplusCONTest)
ACH2EXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevACH2EXPplusCONTest)
ACH2EXPplusCONStats.insert(13, "Diff In Seconds", differenceACH2EXPplusCONTest)
ACH2EXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceACH2EXPplusCONTest)
# ------------------------------Speed
SPH2EXPplusCONAssess = ExperimentalInterventionSpeedAssessmentH2+ControlComparisonSpeedAssessmentH2
SPH2EXPplusCONTest = ExperimentalInterventionSpeedTestingH2+ControlComparisonSpeedTestingH2
# SP H2 EXP plus CON
SPH2EXPplusCONStats = statisticize.ttest(SPH2EXPplusCONAssess,
                                                 SPH2EXPplusCONTest, paired=True, alternative="greater")

meanSPH2EXPplusCONAssess = mean(SPH2EXPplusCONAssess)
pstdevSPH2EXPplusCONAssess = pstdev(SPH2EXPplusCONAssess)

meanSPH2EXPplusCONTest = mean(SPH2EXPplusCONTest)
pstdevSPH2EXPplusCONTest = pstdev(SPH2EXPplusCONTest)

differenceSPH2EXPplusCONTest = meanSPH2EXPplusCONAssess - meanSPH2EXPplusCONTest
fractionalDifferenceSPH2EXPplusCONTest = differenceSPH2EXPplusCONTest/meanSPH2EXPplusCONAssess

SPH2EXPplusCONStats.insert(0, "Name", "Speed across both Conditions (H2)")
SPH2EXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanSPH2EXPplusCONAssess)
SPH2EXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevSPH2EXPplusCONAssess)
SPH2EXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanSPH2EXPplusCONTest)
SPH2EXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevSPH2EXPplusCONTest)
SPH2EXPplusCONStats.insert(13, "Diff In Seconds", differenceSPH2EXPplusCONTest)
SPH2EXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceSPH2EXPplusCONTest)

space = [""]
PerformanceMetricH2EXPplusCONHeader = pd.DataFrame(data={"": space})
PerformanceMetricH2EXPplusCONHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing Within H2 across both Conditions")



# Performance Metric Stats Grouped by PRE- and POST-Intervention across both Hypothesis and conditions
# All 240 datapoints per metric per participant across Both Hypotheses and both conditions
# ------------------------------Resumption
RLBothHypothesesEXPplusCONAssess = ExperimentalInterventionResumptionLagsAssessmentH1+ControlComparisonResumptionLagsAssessmentH1+\
                       ExperimentalInterventionResumptionLagsAssessmentH2+ControlComparisonResumptionLagsAssessmentH2
RLBothHypothesesEXPplusCONTest = ExperimentalInterventionResumptionLagsTestingH1+ControlComparisonResumptionLagsTestingH1+\
                     ExperimentalInterventionResumptionLagsTestingH2+ControlComparisonResumptionLagsTestingH2
# RL Both Hypotheses EXP plus CON
RLBothHypothesesEXPplusCONStats = statisticize.ttest(RLBothHypothesesEXPplusCONAssess,
                                                 RLBothHypothesesEXPplusCONTest, paired=True, alternative="greater")

meanRLBothHypothesesEXPplusCONAssess = mean(RLBothHypothesesEXPplusCONAssess)
pstdevRLBothHypothesesEXPplusCONAssess = pstdev(RLBothHypothesesEXPplusCONAssess)

meanRLBothHypothesesEXPplusCONTest = mean(RLBothHypothesesEXPplusCONTest)
pstdevRLBothHypothesesEXPplusCONTest = pstdev(RLBothHypothesesEXPplusCONTest)

differenceRLBothHypothesesEXPplusCONTest = meanRLBothHypothesesEXPplusCONAssess - meanRLBothHypothesesEXPplusCONTest
fractionalDifferenceRLBothHypothesesEXPplusCONTest = differenceRLBothHypothesesEXPplusCONTest/meanRLBothHypothesesEXPplusCONAssess

RLBothHypothesesEXPplusCONStats.insert(0, "Name", "Pre- vs Post Resumption Lag across both Conditions (BothHypotheses)")
RLBothHypothesesEXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanRLBothHypothesesEXPplusCONAssess)
RLBothHypothesesEXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevRLBothHypothesesEXPplusCONAssess)
RLBothHypothesesEXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanRLBothHypothesesEXPplusCONTest)
RLBothHypothesesEXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevRLBothHypothesesEXPplusCONTest)
RLBothHypothesesEXPplusCONStats.insert(13, "Diff In Seconds", differenceRLBothHypothesesEXPplusCONTest)
RLBothHypothesesEXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceRLBothHypothesesEXPplusCONTest)

# ------------------------------Interruption
ILBothHypothesesEXPplusCONAssess = ExperimentalInterventionInterruptionLagsAssessmentH1+ControlComparisonInterruptionLagsAssessmentH1+\
                                  ExperimentalInterventionInterruptionLagsAssessmentH2+ControlComparisonInterruptionLagsAssessmentH2
ILBothHypothesesEXPplusCONTest = ExperimentalInterventionInterruptionLagsTestingH1+ControlComparisonInterruptionLagsTestingH1+\
                                ExperimentalInterventionInterruptionLagsTestingH2+ControlComparisonInterruptionLagsTestingH2
# IL Both Hypotheses EXP plus CON
ILBothHypothesesEXPplusCONStats = statisticize.ttest(ILBothHypothesesEXPplusCONAssess,
                                                 ILBothHypothesesEXPplusCONTest, paired=True, alternative="greater")

meanILBothHypothesesEXPplusCONAssess = mean(ILBothHypothesesEXPplusCONAssess)
pstdevILBothHypothesesEXPplusCONAssess = pstdev(ILBothHypothesesEXPplusCONAssess)

meanILBothHypothesesEXPplusCONTest = mean(ILBothHypothesesEXPplusCONTest)
pstdevILBothHypothesesEXPplusCONTest = pstdev(ILBothHypothesesEXPplusCONTest)

differenceILBothHypothesesEXPplusCONTest = meanILBothHypothesesEXPplusCONAssess - meanILBothHypothesesEXPplusCONTest
fractionalDifferenceILBothHypothesesEXPplusCONTest = differenceILBothHypothesesEXPplusCONTest/meanILBothHypothesesEXPplusCONAssess

ILBothHypothesesEXPplusCONStats.insert(0, "Name", "Pre- vs Post Interruption Lag across both Conditions (BothHypotheses)")
ILBothHypothesesEXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanILBothHypothesesEXPplusCONAssess)
ILBothHypothesesEXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevILBothHypothesesEXPplusCONAssess)
ILBothHypothesesEXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanILBothHypothesesEXPplusCONTest)
ILBothHypothesesEXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevILBothHypothesesEXPplusCONTest)
ILBothHypothesesEXPplusCONStats.insert(13, "Diff In Seconds", differenceILBothHypothesesEXPplusCONTest)
ILBothHypothesesEXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceILBothHypothesesEXPplusCONTest)

# ------------------------------Accuracy
ACBothHypothesesEXPplusCONAssess = ExperimentalInterventionAccuraciesAssessmentH1+ControlComparisonAccuraciesAssessmentH1+\
                                  ExperimentalInterventionAccuraciesAssessmentH2+ControlComparisonAccuraciesAssessmentH2
ACBothHypothesesEXPplusCONTest = ExperimentalInterventionAccuraciesTestingH1+ControlComparisonAccuraciesTestingH1+\
                                ExperimentalInterventionAccuraciesTestingH2+ControlComparisonAccuraciesTestingH2
# AC Both Hypotheses EXP plus CON
ACBothHypothesesEXPplusCONStats = statisticize.ttest(ACBothHypothesesEXPplusCONAssess,
                                                 ACBothHypothesesEXPplusCONTest, paired=True, alternative="greater")

meanACBothHypothesesEXPplusCONAssess = mean(ACBothHypothesesEXPplusCONAssess)
pstdevACBothHypothesesEXPplusCONAssess = pstdev(ACBothHypothesesEXPplusCONAssess)

meanACBothHypothesesEXPplusCONTest = mean(ACBothHypothesesEXPplusCONTest)
pstdevACBothHypothesesEXPplusCONTest = pstdev(ACBothHypothesesEXPplusCONTest)

differenceACBothHypothesesEXPplusCONTest = meanACBothHypothesesEXPplusCONAssess - meanACBothHypothesesEXPplusCONTest
fractionalDifferenceACBothHypothesesEXPplusCONTest = differenceACBothHypothesesEXPplusCONTest

ACBothHypothesesEXPplusCONStats.insert(0, "Name", "Pre- vs Post Accuracy across both Conditions (BothHypotheses)")
ACBothHypothesesEXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanACBothHypothesesEXPplusCONAssess)
ACBothHypothesesEXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevACBothHypothesesEXPplusCONAssess)
ACBothHypothesesEXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanACBothHypothesesEXPplusCONTest)
ACBothHypothesesEXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevACBothHypothesesEXPplusCONTest)
ACBothHypothesesEXPplusCONStats.insert(13, "Diff In Seconds", differenceACBothHypothesesEXPplusCONTest)
ACBothHypothesesEXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceACBothHypothesesEXPplusCONTest)


# ------------------------------Speed
SPBothHypothesesEXPplusCONAssess = ExperimentalInterventionSpeedAssessmentH1+ControlComparisonSpeedAssessmentH1+\
                                  ExperimentalInterventionSpeedAssessmentH2+ControlComparisonSpeedAssessmentH2
SPBothHypothesesEXPplusCONTest = ExperimentalInterventionSpeedTestingH1+ControlComparisonSpeedTestingH1+\
                                ExperimentalInterventionSpeedTestingH2+ControlComparisonSpeedTestingH2
# SP BothHypotheses EXP plus CON
SPBothHypothesesEXPplusCONStats = statisticize.ttest(SPBothHypothesesEXPplusCONAssess,
                                                 SPBothHypothesesEXPplusCONTest, paired=True, alternative="greater")

meanSPBothHypothesesEXPplusCONAssess = mean(SPBothHypothesesEXPplusCONAssess)
pstdevSPBothHypothesesEXPplusCONAssess = pstdev(SPBothHypothesesEXPplusCONAssess)

meanSPBothHypothesesEXPplusCONTest = mean(SPBothHypothesesEXPplusCONTest)
pstdevSPBothHypothesesEXPplusCONTest = pstdev(SPBothHypothesesEXPplusCONTest)

differenceSPBothHypothesesEXPplusCONTest = meanSPBothHypothesesEXPplusCONAssess - meanSPBothHypothesesEXPplusCONTest
fractionalDifferenceSPBothHypothesesEXPplusCONTest = differenceSPBothHypothesesEXPplusCONTest/meanSPBothHypothesesEXPplusCONAssess

SPBothHypothesesEXPplusCONStats.insert(0, "Name", "Pre- vs Post Speed across both Conditions (BothHypotheses)")
SPBothHypothesesEXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanSPBothHypothesesEXPplusCONAssess)
SPBothHypothesesEXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevSPBothHypothesesEXPplusCONAssess)
SPBothHypothesesEXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanSPBothHypothesesEXPplusCONTest)
SPBothHypothesesEXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevSPBothHypothesesEXPplusCONTest)
SPBothHypothesesEXPplusCONStats.insert(13, "Diff In Seconds", differenceSPBothHypothesesEXPplusCONTest)
SPBothHypothesesEXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceSPBothHypothesesEXPplusCONTest)

space = [""]
PerformanceMetricSPBothHypothesesEXPplusCONStatsEXPplusCONHeader = pd.DataFrame(data={"": space})
PerformanceMetricSPBothHypothesesEXPplusCONStatsEXPplusCONHeader.insert(0, "Name", "Metrics Stats BTW Pre- and Post-Intervention across both Hypotheses and Conditions")





# Path Tracing of Experimental Intervention Resumption Stats
PathTracingExperimentalInterventionResumptionLags = statisticize.ttest(ExperimentalInterventionResumptionLagsAssessmentPathTracing,
                                                 ExperimentalInterventionResumptionLagsTestingPathTracing, paired=True, alternative="greater")

meanExperimentalInterventionResumptionLagsAssessmentPathTracing = mean(ExperimentalInterventionResumptionLagsAssessmentPathTracing)
pstdevExperimentalInterventionResumptionLagsAssessmentPathTracing = pstdev(ExperimentalInterventionResumptionLagsAssessmentPathTracing)

meanExperimentalInterventionResumptionLagsTestingPathTracing = mean(ExperimentalInterventionResumptionLagsTestingPathTracing)
pstdevExperimentalInterventionResumptionLagsTestingPathTracing = pstdev(ExperimentalInterventionResumptionLagsTestingPathTracing)

differenceBtwAssessResumePathTracing = meanExperimentalInterventionResumptionLagsAssessmentPathTracing - meanExperimentalInterventionResumptionLagsTestingPathTracing
fractionalDifferenceBtwAssessResumePathTracing = differenceBtwAssessResumePathTracing/meanExperimentalInterventionResumptionLagsAssessmentPathTracing

PathTracingExperimentalInterventionResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Path Tracing of Experimental Intervention")
PathTracingExperimentalInterventionResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionResumptionLagsAssessmentPathTracing)
PathTracingExperimentalInterventionResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionResumptionLagsAssessmentPathTracing)
PathTracingExperimentalInterventionResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionResumptionLagsTestingPathTracing)
PathTracingExperimentalInterventionResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionResumptionLagsTestingPathTracing)
PathTracingExperimentalInterventionResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResumePathTracing)
PathTracingExperimentalInterventionResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumePathTracing)


# ToH of Experimental Intervention Resumption Stats
ToHExperimentalInterventionResumptionLags = statisticize.ttest(ExperimentalInterventionResumptionLagsAssessmentToH,
                                                 ExperimentalInterventionResumptionLagsTestingToH, paired=True, alternative="greater")

meanExperimentalInterventionResumptionLagsAssessmentToH= mean(ExperimentalInterventionResumptionLagsAssessmentToH)
pstdevExperimentalInterventionResumptionLagsAssessmentToH= pstdev(ExperimentalInterventionResumptionLagsAssessmentToH)

meanExperimentalInterventionResumptionLagsTestingToH= mean(ExperimentalInterventionResumptionLagsTestingToH)
pstdevExperimentalInterventionResumptionLagsTestingToH= pstdev(ExperimentalInterventionResumptionLagsTestingToH)

differenceBtwAssessResumeToH= meanExperimentalInterventionResumptionLagsAssessmentToH- meanExperimentalInterventionResumptionLagsTestingToH
fractionalDifferenceBtwAssessResumeToH= differenceBtwAssessResumeToH/meanExperimentalInterventionResumptionLagsAssessmentToH

ToHExperimentalInterventionResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing ToH of Experimental Intervention")
ToHExperimentalInterventionResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionResumptionLagsAssessmentToH)
ToHExperimentalInterventionResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionResumptionLagsAssessmentToH)
ToHExperimentalInterventionResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionResumptionLagsTestingToH)
ToHExperimentalInterventionResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionResumptionLagsTestingToH)
ToHExperimentalInterventionResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResumeToH)
ToHExperimentalInterventionResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeToH)


# Path Tracing of Experimental Intervention Interruption Stats
PathTracingExperimentalInterventionInterruptionLags = statisticize.ttest(ExperimentalInterventionInterruptionLagsAssessmentPathTracing,
                                                 ExperimentalInterventionInterruptionLagsTestingPathTracing, paired=True, alternative="greater")

meanExperimentalInterventionInterruptionLagsAssessmentPathTracing = mean(ExperimentalInterventionInterruptionLagsAssessmentPathTracing)
pstdevExperimentalInterventionInterruptionLagsAssessmentPathTracing = pstdev(ExperimentalInterventionInterruptionLagsAssessmentPathTracing)

meanExperimentalInterventionInterruptionLagsTestingPathTracing = mean(ExperimentalInterventionInterruptionLagsTestingPathTracing)
pstdevExperimentalInterventionInterruptionLagsTestingPathTracing = pstdev(ExperimentalInterventionInterruptionLagsTestingPathTracing)

differenceBtwAssessAttendPathTracing = meanExperimentalInterventionInterruptionLagsAssessmentPathTracing - meanExperimentalInterventionInterruptionLagsTestingPathTracing
fractionalDifferenceBtwAssessAttendPathTracing = differenceBtwAssessAttendPathTracing/meanExperimentalInterventionInterruptionLagsAssessmentPathTracing

PathTracingExperimentalInterventionInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Path Tracing of Experimental Intervention")
PathTracingExperimentalInterventionInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionInterruptionLagsAssessmentPathTracing)
PathTracingExperimentalInterventionInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionInterruptionLagsAssessmentPathTracing)
PathTracingExperimentalInterventionInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionInterruptionLagsTestingPathTracing)
PathTracingExperimentalInterventionInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionInterruptionLagsTestingPathTracing)
PathTracingExperimentalInterventionInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttendPathTracing)
PathTracingExperimentalInterventionInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendPathTracing)


# ToH of Experimental Intervention Interruption Stats
ToHExperimentalInterventionInterruptionLags = statisticize.ttest(ExperimentalInterventionInterruptionLagsAssessmentToH,
                                                 ExperimentalInterventionInterruptionLagsTestingToH, paired=True, alternative="greater")

meanExperimentalInterventionInterruptionLagsAssessmentToH= mean(ExperimentalInterventionInterruptionLagsAssessmentToH)
pstdevExperimentalInterventionInterruptionLagsAssessmentToH= pstdev(ExperimentalInterventionInterruptionLagsAssessmentToH)

meanExperimentalInterventionInterruptionLagsTestingToH= mean(ExperimentalInterventionInterruptionLagsTestingToH)
pstdevExperimentalInterventionInterruptionLagsTestingToH= pstdev(ExperimentalInterventionInterruptionLagsTestingToH)

differenceBtwAssessAttendToH= meanExperimentalInterventionInterruptionLagsAssessmentToH- meanExperimentalInterventionInterruptionLagsTestingToH
fractionalDifferenceBtwAssessAttendToH= differenceBtwAssessAttendToH/meanExperimentalInterventionInterruptionLagsAssessmentToH

ToHExperimentalInterventionInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing ToH of Experimental Intervention")
ToHExperimentalInterventionInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionInterruptionLagsAssessmentToH)
ToHExperimentalInterventionInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionInterruptionLagsAssessmentToH)
ToHExperimentalInterventionInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionInterruptionLagsTestingToH)
ToHExperimentalInterventionInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionInterruptionLagsTestingToH)
ToHExperimentalInterventionInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttendToH)
ToHExperimentalInterventionInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendToH)


# Path Tracing of Experimental Intervention Accuracy Stats
PathTracingExperimentalInterventionAccuracies = statisticize.ttest(ExperimentalInterventionAccuraciesAssessmentPathTracing,
                                                 ExperimentalInterventionAccuraciesTestingPathTracing, paired=True, alternative="greater")

meanExperimentalInterventionAccuraciesAssessmentPathTracing = mean(ExperimentalInterventionAccuraciesAssessmentPathTracing)
pstdevExperimentalInterventionAccuraciesAssessmentPathTracing = pstdev(ExperimentalInterventionAccuraciesAssessmentPathTracing)

meanExperimentalInterventionAccuraciesTestingPathTracing = mean(ExperimentalInterventionAccuraciesTestingPathTracing)
pstdevExperimentalInterventionAccuraciesTestingPathTracing = pstdev(ExperimentalInterventionAccuraciesTestingPathTracing)

differenceBtwAssessAccuraciesPathTracing = meanExperimentalInterventionAccuraciesAssessmentPathTracing - meanExperimentalInterventionAccuraciesTestingPathTracing
fractionalDifferenceBtwAssessAccuraciesPathTracing = differenceBtwAssessAccuraciesPathTracing

PathTracingExperimentalInterventionAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Path Tracing of Experimental Intervention")
PathTracingExperimentalInterventionAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionAccuraciesAssessmentPathTracing)
PathTracingExperimentalInterventionAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionAccuraciesAssessmentPathTracing)
PathTracingExperimentalInterventionAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionAccuraciesTestingPathTracing)
PathTracingExperimentalInterventionAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionAccuraciesTestingPathTracing)
PathTracingExperimentalInterventionAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesPathTracing)
PathTracingExperimentalInterventionAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesPathTracing)


# ToH of Experimental Intervention Accuracy Stats
ToHExperimentalInterventionAccuracies = statisticize.ttest(ExperimentalInterventionAccuraciesAssessmentToH,
                                                 ExperimentalInterventionAccuraciesTestingToH, paired=True, alternative="greater")

meanExperimentalInterventionAccuraciesAssessmentToH= mean(ExperimentalInterventionAccuraciesAssessmentToH)
pstdevExperimentalInterventionAccuraciesAssessmentToH= pstdev(ExperimentalInterventionAccuraciesAssessmentToH)

meanExperimentalInterventionAccuraciesTestingToH= mean(ExperimentalInterventionAccuraciesTestingToH)
pstdevExperimentalInterventionAccuraciesTestingToH= pstdev(ExperimentalInterventionAccuraciesTestingToH)

differenceBtwAssessAccuraciesToH= meanExperimentalInterventionAccuraciesAssessmentToH- meanExperimentalInterventionAccuraciesTestingToH
fractionalDifferenceBtwAssessAccuraciesToH= differenceBtwAssessAccuraciesToH

ToHExperimentalInterventionAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing ToH of Experimental Intervention")
ToHExperimentalInterventionAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionAccuraciesAssessmentToH)
ToHExperimentalInterventionAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionAccuraciesAssessmentToH)
ToHExperimentalInterventionAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionAccuraciesTestingToH)
ToHExperimentalInterventionAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionAccuraciesTestingToH)
ToHExperimentalInterventionAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesToH)
ToHExperimentalInterventionAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesToH)


# Path Tracing of Experimental Intervention Speed Stats
PathTracingExperimentalInterventionSpeed = statisticize.ttest(ExperimentalInterventionSpeedAssessmentPathTracing,
                                                 ExperimentalInterventionSpeedTestingPathTracing, paired=True, alternative="greater")

meanExperimentalInterventionSpeedAssessmentPathTracing = mean(ExperimentalInterventionSpeedAssessmentPathTracing)
pstdevExperimentalInterventionSpeedAssessmentPathTracing = pstdev(ExperimentalInterventionSpeedAssessmentPathTracing)

meanExperimentalInterventionSpeedTestingPathTracing = mean(ExperimentalInterventionSpeedTestingPathTracing)
pstdevExperimentalInterventionSpeedTestingPathTracing = pstdev(ExperimentalInterventionSpeedTestingPathTracing)

differenceBtwAssessSpeedsPathTracing = meanExperimentalInterventionSpeedAssessmentPathTracing - meanExperimentalInterventionSpeedTestingPathTracing
fractionalDifferenceBtwAssessSpeedsPathTracing = differenceBtwAssessSpeedsPathTracing/meanExperimentalInterventionSpeedAssessmentPathTracing

PathTracingExperimentalInterventionSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Path Tracing of Experimental Intervention")
PathTracingExperimentalInterventionSpeed.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionSpeedAssessmentPathTracing)
PathTracingExperimentalInterventionSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionSpeedAssessmentPathTracing)
PathTracingExperimentalInterventionSpeed.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionSpeedTestingPathTracing)
PathTracingExperimentalInterventionSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionSpeedTestingPathTracing)
PathTracingExperimentalInterventionSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsPathTracing)
PathTracingExperimentalInterventionSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsPathTracing)

# ToH of Experimental Intervention Speed Stats
ToHExperimentalInterventionSpeed = statisticize.ttest(ExperimentalInterventionSpeedAssessmentToH,
                                                 ExperimentalInterventionSpeedTestingToH, paired=True, alternative="greater")

meanExperimentalInterventionSpeedAssessmentToH= mean(ExperimentalInterventionSpeedAssessmentToH)
pstdevExperimentalInterventionSpeedAssessmentToH= pstdev(ExperimentalInterventionSpeedAssessmentToH)

meanExperimentalInterventionSpeedTestingToH= mean(ExperimentalInterventionSpeedTestingToH)
pstdevExperimentalInterventionSpeedTestingToH= pstdev(ExperimentalInterventionSpeedTestingToH)

differenceBtwAssessSpeedsToH= meanExperimentalInterventionSpeedAssessmentToH- meanExperimentalInterventionSpeedTestingToH
fractionalDifferenceBtwAssessSpeedsToH= differenceBtwAssessSpeedsToH/meanExperimentalInterventionSpeedAssessmentToH

ToHExperimentalInterventionSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing ToH of Experimental Intervention")
ToHExperimentalInterventionSpeed.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionSpeedAssessmentToH)
ToHExperimentalInterventionSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionSpeedAssessmentToH)
ToHExperimentalInterventionSpeed.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionSpeedTestingToH)
ToHExperimentalInterventionSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionSpeedTestingToH)
ToHExperimentalInterventionSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsToH)
ToHExperimentalInterventionSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsToH)



# Control Comparison Stats
# Path Tracing of Control Comparison Resumption Stats
PathTracingControlComparisonResumptionLags = statisticize.ttest(ControlComparisonResumptionLagsAssessmentPathTracing,
                                                 ControlComparisonResumptionLagsTestingPathTracing, paired=True, alternative="greater")

meanControlComparisonResumptionLagsAssessmentPathTracing = mean(ControlComparisonResumptionLagsAssessmentPathTracing)
pstdevControlComparisonResumptionLagsAssessmentPathTracing = pstdev(ControlComparisonResumptionLagsAssessmentPathTracing)

meanControlComparisonResumptionLagsTestingPathTracing = mean(ControlComparisonResumptionLagsTestingPathTracing)
pstdevControlComparisonResumptionLagsTestingPathTracing = pstdev(ControlComparisonResumptionLagsTestingPathTracing)

differenceBtwAssessResumeControlPathTracing = meanControlComparisonResumptionLagsAssessmentPathTracing - meanControlComparisonResumptionLagsTestingPathTracing
fractionalDifferenceBtwAssessResumeControlPathTracing = differenceBtwAssessResumeControlPathTracing/meanControlComparisonResumptionLagsAssessmentPathTracing

PathTracingControlComparisonResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Path Tracing Control Comparison")
PathTracingControlComparisonResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonResumptionLagsAssessmentPathTracing)
PathTracingControlComparisonResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonResumptionLagsAssessmentPathTracing)
PathTracingControlComparisonResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonResumptionLagsTestingPathTracing)
PathTracingControlComparisonResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonResumptionLagsTestingPathTracing)
PathTracingControlComparisonResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResumeControlPathTracing)
PathTracingControlComparisonResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeControlPathTracing)


# To Hof Control Comparison Resumption Stats
ToHControlComparisonResumptionLags = statisticize.ttest(ControlComparisonResumptionLagsAssessmentToH,
                                                 ControlComparisonResumptionLagsTestingToH, paired=True, alternative="greater")

meanControlComparisonResumptionLagsAssessmentToH= mean(ControlComparisonResumptionLagsAssessmentToH)
pstdevControlComparisonResumptionLagsAssessmentToH= pstdev(ControlComparisonResumptionLagsAssessmentToH)

meanControlComparisonResumptionLagsTestingToH= mean(ControlComparisonResumptionLagsTestingToH)
pstdevControlComparisonResumptionLagsTestingToH= pstdev(ControlComparisonResumptionLagsTestingToH)

differenceBtwAssessResumeControlToH= meanControlComparisonResumptionLagsAssessmentToH- meanControlComparisonResumptionLagsTestingToH
fractionalDifferenceBtwAssessResumeControlToH= differenceBtwAssessResumeControlToH/meanControlComparisonResumptionLagsAssessmentToH

ToHControlComparisonResumptionLags.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing ToH Control Comparison")
ToHControlComparisonResumptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonResumptionLagsAssessmentToH)
ToHControlComparisonResumptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonResumptionLagsAssessmentToH)
ToHControlComparisonResumptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonResumptionLagsTestingToH)
ToHControlComparisonResumptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonResumptionLagsTestingToH)
ToHControlComparisonResumptionLags.insert(13, "Diff In Seconds", differenceBtwAssessResumeControlToH)
ToHControlComparisonResumptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeControlToH)


# Path Tracing of Control Comparison Interruption Stats
PathTracingControlComparisonInterruptionLags = statisticize.ttest(ControlComparisonInterruptionLagsAssessmentPathTracing,
                                                 ControlComparisonInterruptionLagsTestingPathTracing, paired=True, alternative="greater")

meanControlComparisonInterruptionLagsAssessmentPathTracing = mean(ControlComparisonInterruptionLagsAssessmentPathTracing)
pstdevControlComparisonInterruptionLagsAssessmentPathTracing = pstdev(ControlComparisonInterruptionLagsAssessmentPathTracing)

meanControlComparisonInterruptionLagsTestingPathTracing = mean(ControlComparisonInterruptionLagsTestingPathTracing)
pstdevControlComparisonInterruptionLagsTestingPathTracing = pstdev(ControlComparisonInterruptionLagsTestingPathTracing)

differenceBtwAssessAttendControlPathTracing = meanControlComparisonInterruptionLagsAssessmentPathTracing - meanControlComparisonInterruptionLagsTestingPathTracing
fractionalDifferenceBtwAssessAttendControlPathTracing = differenceBtwAssessAttendControlPathTracing/meanControlComparisonInterruptionLagsAssessmentPathTracing

PathTracingControlComparisonInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Path Tracing Control Comparison")
PathTracingControlComparisonInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonInterruptionLagsAssessmentPathTracing)
PathTracingControlComparisonInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonInterruptionLagsAssessmentPathTracing)
PathTracingControlComparisonInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonInterruptionLagsTestingPathTracing)
PathTracingControlComparisonInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonInterruptionLagsTestingPathTracing)
PathTracingControlComparisonInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttendControlPathTracing)
PathTracingControlComparisonInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendControlPathTracing)


# ToH of Control Comparison Interruption Stats
ToHControlComparisonInterruptionLags = statisticize.ttest(ControlComparisonInterruptionLagsAssessmentToH,
                                                 ControlComparisonInterruptionLagsTestingToH, paired=True, alternative="greater")

meanControlComparisonInterruptionLagsAssessmentToH= mean(ControlComparisonInterruptionLagsAssessmentToH)
pstdevControlComparisonInterruptionLagsAssessmentToH= pstdev(ControlComparisonInterruptionLagsAssessmentToH)

meanControlComparisonInterruptionLagsTestingToH= mean(ControlComparisonInterruptionLagsTestingToH)
pstdevControlComparisonInterruptionLagsTestingToH= pstdev(ControlComparisonInterruptionLagsTestingToH)

differenceBtwAssessAttendControlToH= meanControlComparisonInterruptionLagsAssessmentToH- meanControlComparisonInterruptionLagsTestingToH
fractionalDifferenceBtwAssessAttendControlToH= differenceBtwAssessAttendControlToH/meanControlComparisonInterruptionLagsAssessmentToH

ToHControlComparisonInterruptionLags.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing ToH Control Comparison")
ToHControlComparisonInterruptionLags.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonInterruptionLagsAssessmentToH)
ToHControlComparisonInterruptionLags.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonInterruptionLagsAssessmentToH)
ToHControlComparisonInterruptionLags.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonInterruptionLagsTestingToH)
ToHControlComparisonInterruptionLags.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonInterruptionLagsTestingToH)
ToHControlComparisonInterruptionLags.insert(13, "Diff In Seconds", differenceBtwAssessAttendControlToH)
ToHControlComparisonInterruptionLags.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendControlToH)


# Path Tracing of Control Comparison Accuracy Stats
PathTracingControlComparisonAccuracies = statisticize.ttest(ControlComparisonAccuraciesAssessmentPathTracing,
                                                 ControlComparisonAccuraciesTestingPathTracing, paired=True, alternative="greater")

meanControlComparisonAccuraciesAssessmentPathTracing = mean(ControlComparisonAccuraciesAssessmentPathTracing)
pstdevControlComparisonAccuraciesAssessmentPathTracing = pstdev(ControlComparisonAccuraciesAssessmentPathTracing)

meanControlComparisonAccuraciesTestingPathTracing = mean(ControlComparisonAccuraciesTestingPathTracing)
pstdevControlComparisonAccuraciesTestingPathTracing = pstdev(ControlComparisonAccuraciesTestingPathTracing)

differenceBtwAssessAccuraciesControlPathTracing = meanControlComparisonAccuraciesAssessmentPathTracing - meanControlComparisonAccuraciesTestingPathTracing
fractionalDifferenceBtwAssessAccuraciesControlPathTracing = differenceBtwAssessAccuraciesControlPathTracing

PathTracingControlComparisonAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Path Tracing Control Comparison")
PathTracingControlComparisonAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonAccuraciesAssessmentPathTracing)
PathTracingControlComparisonAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonAccuraciesAssessmentPathTracing)
PathTracingControlComparisonAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonAccuraciesTestingPathTracing)
PathTracingControlComparisonAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonAccuraciesTestingPathTracing)
PathTracingControlComparisonAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesControlPathTracing)
PathTracingControlComparisonAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesControlPathTracing)


# ToH of Control Comparison Accuracy Stats
ToHControlComparisonAccuracies = statisticize.ttest(ControlComparisonAccuraciesAssessmentToH,
                                                 ControlComparisonAccuraciesTestingToH, paired=True, alternative="greater")

meanControlComparisonAccuraciesAssessmentToH= mean(ControlComparisonAccuraciesAssessmentToH)
pstdevControlComparisonAccuraciesAssessmentToH= pstdev(ControlComparisonAccuraciesAssessmentToH)

meanControlComparisonAccuraciesTestingToH= mean(ControlComparisonAccuraciesTestingToH)
pstdevControlComparisonAccuraciesTestingToH= pstdev(ControlComparisonAccuraciesTestingToH)

differenceBtwAssessAccuraciesControlToH= meanControlComparisonAccuraciesAssessmentToH- meanControlComparisonAccuraciesTestingToH
fractionalDifferenceBtwAssessAccuraciesControlToH= differenceBtwAssessAccuraciesControlToH

ToHControlComparisonAccuracies.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing ToH Control Comparison")
ToHControlComparisonAccuracies.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonAccuraciesAssessmentToH)
ToHControlComparisonAccuracies.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonAccuraciesAssessmentToH)
ToHControlComparisonAccuracies.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonAccuraciesTestingToH)
ToHControlComparisonAccuracies.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonAccuraciesTestingToH)
ToHControlComparisonAccuracies.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesControlToH)
ToHControlComparisonAccuracies.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesControlToH)


# Path Tracing of Control Comparison Speed Stats
PathTracingControlComparisonSpeed = statisticize.ttest(ControlComparisonSpeedAssessmentPathTracing,
                                                 ControlComparisonSpeedTestingPathTracing, paired=True, alternative="greater")

meanControlComparisonSpeedAssessmentPathTracing = mean(ControlComparisonSpeedAssessmentPathTracing)
pstdevControlComparisonSpeedAssessmentPathTracing = pstdev(ControlComparisonSpeedAssessmentPathTracing)

meanControlComparisonSpeedTestingPathTracing = mean(ControlComparisonSpeedTestingPathTracing)
pstdevControlComparisonSpeedTestingPathTracing = pstdev(ControlComparisonSpeedTestingPathTracing)

differenceBtwAssessSpeedsControlPathTracing = meanControlComparisonSpeedAssessmentPathTracing - meanControlComparisonSpeedTestingPathTracing
fractionalDifferenceBtwAssessSpeedsControlPathTracing = differenceBtwAssessSpeedsControlPathTracing/meanControlComparisonSpeedAssessmentPathTracing

PathTracingControlComparisonSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Path Tracing Control Comparison")
PathTracingControlComparisonSpeed.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonSpeedAssessmentPathTracing)
PathTracingControlComparisonSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonSpeedAssessmentPathTracing)
PathTracingControlComparisonSpeed.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonSpeedTestingPathTracing)
PathTracingControlComparisonSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonSpeedTestingPathTracing)
PathTracingControlComparisonSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsControlPathTracing)
PathTracingControlComparisonSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsControlPathTracing)


# ToH of Control Comparison Speed Stats
ToHControlComparisonSpeed = statisticize.ttest(ControlComparisonSpeedAssessmentToH,
                                                 ControlComparisonSpeedTestingToH, paired=True, alternative="greater")

meanControlComparisonSpeedAssessmentToH= mean(ControlComparisonSpeedAssessmentToH)
pstdevControlComparisonSpeedAssessmentToH= pstdev(ControlComparisonSpeedAssessmentToH)

meanControlComparisonSpeedTestingToH= mean(ControlComparisonSpeedTestingToH)
pstdevControlComparisonSpeedTestingToH= pstdev(ControlComparisonSpeedTestingToH)

differenceBtwAssessSpeedsControlToH= meanControlComparisonSpeedAssessmentToH- meanControlComparisonSpeedTestingToH
fractionalDifferenceBtwAssessSpeedsControlToH= differenceBtwAssessSpeedsControlToH/meanControlComparisonSpeedAssessmentToH

ToHControlComparisonSpeed.insert(0, "Name", "Speed's Stats BTW Assessment and Testing ToH Control Comparison")
ToHControlComparisonSpeed.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonSpeedAssessmentToH)
ToHControlComparisonSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonSpeedAssessmentToH)
ToHControlComparisonSpeed.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonSpeedTestingToH)
ToHControlComparisonSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonSpeedTestingToH)
ToHControlComparisonSpeed.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsControlToH)
ToHControlComparisonSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsControlToH)



# Stats of Task Type Comparisons
PTTDiffResumeAssess = ExperimentalInterventionResumptionLagsAssessmentPathTracing+ControlComparisonResumptionLagsAssessmentPathTracing
PTTDiffResumeTest = ExperimentalInterventionResumptionLagsTestingPathTracing+ControlComparisonResumptionLagsTestingPathTracing
differenceBtwAssessNTestPTTResume = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffResumeAssess, PTTDiffResumeTest)]

ToHDiffResumeAssess = ExperimentalInterventionResumptionLagsAssessmentToH+ControlComparisonResumptionLagsAssessmentToH
ToHDiffResumeTest = ExperimentalInterventionResumptionLagsTestingToH+ControlComparisonResumptionLagsTestingToH
differenceBtwAssessNTestToHResume = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffResumeAssess, ToHDiffResumeTest)]

# Task Type Comparisons for Resumption
TaskTypeComparisonResume = statisticize.ttest(differenceBtwAssessNTestPTTResume,
                                                 differenceBtwAssessNTestToHResume, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTResume= mean(differenceBtwAssessNTestPTTResume)
pstdevDifferenceBtwAssessNTestPTTResume= pstdev(differenceBtwAssessNTestPTTResume)

meanDifferenceBtwAssessNTestToHResume= mean(differenceBtwAssessNTestToHResume)
pstdevDifferenceBtwAssessNTestToHResume= pstdev(differenceBtwAssessNTestToHResume)

differenceTaskTypeComparisonResume= meanDifferenceBtwAssessNTestPTTResume- meanDifferenceBtwAssessNTestToHResume
fractionalDifferenceTaskTypeComparisonResume= differenceTaskTypeComparisonResume/meanDifferenceBtwAssessNTestPTTResume

TaskTypeComparisonResume.insert(0, "Name", "Task Type Comparisons for Resumption")
TaskTypeComparisonResume.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTResume)
TaskTypeComparisonResume.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTResume)
TaskTypeComparisonResume.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHResume)
TaskTypeComparisonResume.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHResume)
TaskTypeComparisonResume.insert(13, "Diff In Seconds", differenceTaskTypeComparisonResume)
TaskTypeComparisonResume.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonResume)


PTTDiffAttendAssess = ExperimentalInterventionInterruptionLagsAssessmentPathTracing+ControlComparisonInterruptionLagsAssessmentPathTracing
PTTDiffAttendTest = ExperimentalInterventionInterruptionLagsTestingPathTracing+ControlComparisonInterruptionLagsTestingPathTracing
differenceBtwAssessNTestPTTAttend = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffAttendAssess, PTTDiffAttendTest)]

ToHDiffAttendAssess = ExperimentalInterventionInterruptionLagsAssessmentToH+ControlComparisonInterruptionLagsAssessmentToH
ToHDiffAttendATest = ExperimentalInterventionInterruptionLagsTestingToH+ControlComparisonInterruptionLagsTestingToH
differenceBtwAssessNTestToHAttend = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffAttendAssess, PTTDiffAttendTest)]

# Task Type Comparisons for Interruption
TaskTypeComparisonAttend = statisticize.ttest(differenceBtwAssessNTestPTTAttend,
                                                 differenceBtwAssessNTestToHAttend, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTAttend= mean(differenceBtwAssessNTestPTTAttend)
pstdevDifferenceBtwAssessNTestPTTAttend= pstdev(differenceBtwAssessNTestPTTAttend)

meanDifferenceBtwAssessNTestToHAttend= mean(differenceBtwAssessNTestToHAttend)
pstdevDifferenceBtwAssessNTestToHAttend= pstdev(differenceBtwAssessNTestToHAttend)

differenceTaskTypeComparisonAttend= meanDifferenceBtwAssessNTestPTTAttend- meanDifferenceBtwAssessNTestToHAttend
fractionalDifferenceTaskTypeComparisonAttend= differenceTaskTypeComparisonAttend/meanDifferenceBtwAssessNTestPTTAttend

TaskTypeComparisonAttend.insert(0, "Name", "Task Type Comparisons for Interruption")
TaskTypeComparisonAttend.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTAttend)
TaskTypeComparisonAttend.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTAttend)
TaskTypeComparisonAttend.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHAttend)
TaskTypeComparisonAttend.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHAttend)
TaskTypeComparisonAttend.insert(13, "Diff In Seconds", differenceTaskTypeComparisonAttend)
TaskTypeComparisonAttend.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonAttend)


PTTDiffAccuracyAssess = ExperimentalInterventionAccuraciesAssessmentPathTracing+ControlComparisonAccuraciesAssessmentPathTracing
PTTDiffAccuracyTest = ExperimentalInterventionAccuraciesTestingPathTracing+ControlComparisonAccuraciesTestingPathTracing
differenceBtwAssessNTestPTTAccuracy = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffAccuracyAssess, PTTDiffAccuracyTest)]

ToHDiffAccuracyAssess = ExperimentalInterventionAccuraciesAssessmentToH+ControlComparisonAccuraciesAssessmentToH
ToHDiffAccuracyTest = ExperimentalInterventionAccuraciesTestingToH+ControlComparisonAccuraciesTestingToH
differenceBtwAssessNTestToHAccuracy = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffAccuracyAssess, ToHDiffAccuracyTest)]

# Task Type Comparisons for Accuracy
TaskTypeComparisonAccuracy = statisticize.ttest(differenceBtwAssessNTestPTTAccuracy,
                                                 differenceBtwAssessNTestToHAccuracy, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTAccuracy= mean(differenceBtwAssessNTestPTTAccuracy)
pstdevDifferenceBtwAssessNTestPTTAccuracy= pstdev(differenceBtwAssessNTestPTTAccuracy)

meanDifferenceBtwAssessNTestToHAccuracy= mean(differenceBtwAssessNTestToHAccuracy)
pstdevDifferenceBtwAssessNTestToHAccuracy= pstdev(differenceBtwAssessNTestToHAccuracy)

differenceTaskTypeComparisonAccuracy= meanDifferenceBtwAssessNTestPTTAccuracy- meanDifferenceBtwAssessNTestToHAccuracy
fractionalDifferenceTaskTypeComparisonAccuracy= differenceTaskTypeComparisonAccuracy#/meanDifferenceBtwAssessNTestPTTAccuracy

TaskTypeComparisonAccuracy.insert(0, "Name", "Task Type Comparisons for Accuracy")
TaskTypeComparisonAccuracy.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTAccuracy)
TaskTypeComparisonAccuracy.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTAccuracy)
TaskTypeComparisonAccuracy.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHAccuracy)
TaskTypeComparisonAccuracy.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHAccuracy)
TaskTypeComparisonAccuracy.insert(13, "Diff In Seconds", differenceTaskTypeComparisonAccuracy)
TaskTypeComparisonAccuracy.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonAccuracy)


PTTDiffSpeedAssess = ExperimentalInterventionSpeedAssessmentPathTracing+ControlComparisonSpeedAssessmentPathTracing
PTTDiffSpeedTest = ExperimentalInterventionSpeedTestingPathTracing+ControlComparisonSpeedTestingPathTracing
differenceBtwAssessNTestPTTSpeed = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffSpeedAssess, PTTDiffSpeedTest)]

ToHDiffSpeedAssess = ExperimentalInterventionSpeedAssessmentToH+ControlComparisonSpeedAssessmentToH
ToHDiffSpeedTest = ExperimentalInterventionSpeedTestingToH+ControlComparisonSpeedTestingToH
differenceBtwAssessNTestToHSpeed = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffSpeedAssess, ToHDiffSpeedTest)]

# Task Type Comparisons for Speed
TaskTypeComparisonSpeed = statisticize.ttest(differenceBtwAssessNTestPTTSpeed,
                                                 differenceBtwAssessNTestToHSpeed, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTSpeed= mean(differenceBtwAssessNTestPTTSpeed)
pstdevDifferenceBtwAssessNTestPTTSpeed= pstdev(differenceBtwAssessNTestPTTSpeed)

meanDifferenceBtwAssessNTestToHSpeed= mean(differenceBtwAssessNTestToHSpeed)
pstdevDifferenceBtwAssessNTestToHSpeed= pstdev(differenceBtwAssessNTestToHSpeed)

differenceTaskTypeComparisonSpeed= meanDifferenceBtwAssessNTestPTTSpeed- meanDifferenceBtwAssessNTestToHSpeed
fractionalDifferenceTaskTypeComparisonSpeed= differenceTaskTypeComparisonSpeed/meanDifferenceBtwAssessNTestPTTSpeed

TaskTypeComparisonSpeed.insert(0, "Name", "Task Type Comparisons for Speed")
TaskTypeComparisonSpeed.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTSpeed)
TaskTypeComparisonSpeed.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTSpeed)
TaskTypeComparisonSpeed.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHSpeed)
TaskTypeComparisonSpeed.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHSpeed)
TaskTypeComparisonSpeed.insert(13, "Diff In Seconds", differenceTaskTypeComparisonSpeed)
TaskTypeComparisonSpeed.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonSpeed)

space = [""]
TaskTypeComparisonsforPerformanceMetricsHeader = pd.DataFrame(data={"": space})
TaskTypeComparisonsforPerformanceMetricsHeader.insert(0, "Name", "Task Type Comparisons for Each performance Metric")




# Stats of Task Type Comparisons Experimental Intervention
PTTDiffResumeEXPAssess = ExperimentalInterventionResumptionLagsAssessmentPathTracing
PTTDiffResumeEXPTest = ExperimentalInterventionResumptionLagsTestingPathTracing
differenceBtwAssessNTestPTTResumeEXP = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffResumeEXPAssess, PTTDiffResumeEXPTest)]

ToHDiffResumeEXPAssess = ExperimentalInterventionResumptionLagsAssessmentToH
ToHDiffResumeEXPTest = ExperimentalInterventionResumptionLagsTestingToH
differenceBtwAssessNTestToHResumeEXP = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffResumeEXPAssess, ToHDiffResumeEXPTest)]

# Task Type Comparisons for Resumption
TaskTypeComparisonResumeEXP = statisticize.ttest(differenceBtwAssessNTestPTTResumeEXP,
                                                 differenceBtwAssessNTestToHResumeEXP, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTResumeEXP= mean(differenceBtwAssessNTestPTTResumeEXP)
pstdevDifferenceBtwAssessNTestPTTResumeEXP= pstdev(differenceBtwAssessNTestPTTResumeEXP)

meanDifferenceBtwAssessNTestToHResumeEXP= mean(differenceBtwAssessNTestToHResumeEXP)
pstdevDifferenceBtwAssessNTestToHResumeEXP= pstdev(differenceBtwAssessNTestToHResumeEXP)

differenceTaskTypeComparisonResumeEXP= meanDifferenceBtwAssessNTestPTTResumeEXP- meanDifferenceBtwAssessNTestToHResumeEXP
fractionalDifferenceTaskTypeComparisonResumeEXP= differenceTaskTypeComparisonResumeEXP/meanDifferenceBtwAssessNTestPTTResumeEXP

TaskTypeComparisonResumeEXP.insert(0, "Name", "Task Type Comparisons for Resumption EXP")
TaskTypeComparisonResumeEXP.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTResumeEXP)
TaskTypeComparisonResumeEXP.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTResumeEXP)
TaskTypeComparisonResumeEXP.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHResumeEXP)
TaskTypeComparisonResumeEXP.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHResumeEXP)
TaskTypeComparisonResumeEXP.insert(13, "Diff In Seconds", differenceTaskTypeComparisonResumeEXP)
TaskTypeComparisonResumeEXP.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonResumeEXP)


PTTDiffAttendEXPAssess = ExperimentalInterventionInterruptionLagsAssessmentPathTracing
PTTDiffAttendEXPTest = ExperimentalInterventionInterruptionLagsTestingPathTracing
differenceBtwAssessNTestPTTAttendEXP = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffAttendEXPAssess, PTTDiffAttendEXPTest)]

ToHDiffAttendEXPAssess = ExperimentalInterventionInterruptionLagsAssessmentToH
ToHDiffAttendEXPTest = ExperimentalInterventionInterruptionLagsTestingToH
differenceBtwAssessNTestToHAttendEXP = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffAttendEXPAssess, ToHDiffAttendEXPTest)]

# Task Type Comparisons for Interruption
TaskTypeComparisonAttendEXP = statisticize.ttest(differenceBtwAssessNTestPTTAttendEXP,
                                                 differenceBtwAssessNTestToHAttendEXP, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTAttendEXP= mean(differenceBtwAssessNTestPTTAttendEXP)
pstdevDifferenceBtwAssessNTestPTTAttendEXP= pstdev(differenceBtwAssessNTestPTTAttendEXP)

meanDifferenceBtwAssessNTestToHAttendEXP= mean(differenceBtwAssessNTestToHAttendEXP)
pstdevDifferenceBtwAssessNTestToHAttendEXP= pstdev(differenceBtwAssessNTestToHAttendEXP)

differenceTaskTypeComparisonAttendEXP= meanDifferenceBtwAssessNTestPTTAttendEXP- meanDifferenceBtwAssessNTestToHAttendEXP
fractionalDifferenceTaskTypeComparisonAttendEXP= differenceTaskTypeComparisonAttendEXP/meanDifferenceBtwAssessNTestPTTAttendEXP

TaskTypeComparisonAttendEXP.insert(0, "Name", "Task Type Comparisons for Interruption EXP")
TaskTypeComparisonAttendEXP.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTAttendEXP)
TaskTypeComparisonAttendEXP.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTAttendEXP)
TaskTypeComparisonAttendEXP.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHAttendEXP)
TaskTypeComparisonAttendEXP.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHAttendEXP)
TaskTypeComparisonAttendEXP.insert(13, "Diff In Seconds", differenceTaskTypeComparisonAttendEXP)
TaskTypeComparisonAttendEXP.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonAttendEXP)


PTTDiffAccuracyEXPAssess = ExperimentalInterventionAccuraciesAssessmentPathTracing
PTTDiffAccuracyEXPTest = ExperimentalInterventionAccuraciesTestingPathTracing
differenceBtwAssessNTestPTTAccuracyEXP = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffAccuracyEXPAssess, PTTDiffAccuracyEXPTest)]

ToHDiffAccuracyEXPAssess = ExperimentalInterventionAccuraciesAssessmentToH
ToHDiffAccuracyEXPTest = ExperimentalInterventionAccuraciesTestingToH
differenceBtwAssessNTestToHAccuracyEXP = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffAccuracyEXPAssess, ToHDiffAccuracyEXPTest)]

# Task Type Comparisons for AccuracyEXP
TaskTypeComparisonAccuracyEXP = statisticize.ttest(differenceBtwAssessNTestPTTAccuracyEXP,
                                                 differenceBtwAssessNTestToHAccuracyEXP, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTAccuracyEXP= mean(differenceBtwAssessNTestPTTAccuracyEXP)
pstdevDifferenceBtwAssessNTestPTTAccuracyEXP= pstdev(differenceBtwAssessNTestPTTAccuracyEXP)

meanDifferenceBtwAssessNTestToHAccuracyEXP= mean(differenceBtwAssessNTestToHAccuracyEXP)
pstdevDifferenceBtwAssessNTestToHAccuracyEXP= pstdev(differenceBtwAssessNTestToHAccuracyEXP)

differenceTaskTypeComparisonAccuracyEXP= meanDifferenceBtwAssessNTestPTTAccuracyEXP- meanDifferenceBtwAssessNTestToHAccuracyEXP
fractionalDifferenceTaskTypeComparisonAccuracyEXP= differenceTaskTypeComparisonAccuracyEXP

TaskTypeComparisonAccuracyEXP.insert(0, "Name", "Task Type Comparisons for Accuracy EXP")
TaskTypeComparisonAccuracyEXP.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTAccuracyEXP)
TaskTypeComparisonAccuracyEXP.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTAccuracyEXP)
TaskTypeComparisonAccuracyEXP.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHAccuracyEXP)
TaskTypeComparisonAccuracyEXP.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHAccuracyEXP)
TaskTypeComparisonAccuracyEXP.insert(13, "Diff In Seconds", differenceTaskTypeComparisonAccuracyEXP)
TaskTypeComparisonAccuracyEXP.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonAccuracyEXP)


PTTDiffSpeedEXPAssess = ExperimentalInterventionSpeedAssessmentPathTracing
PTTDiffSpeedEXPTest = ExperimentalInterventionSpeedTestingPathTracing
differenceBtwAssessNTestPTTSpeedEXP = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffSpeedEXPAssess, PTTDiffSpeedEXPTest)]

ToHDiffSpeedEXPAssess = ExperimentalInterventionSpeedAssessmentToH
ToHDiffSpeedEXPTest = ExperimentalInterventionSpeedTestingToH
differenceBtwAssessNTestToHSpeedEXP = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffSpeedEXPAssess, ToHDiffSpeedEXPTest)]

# Task Type Comparisons for Speed
TaskTypeComparisonSpeedEXP = statisticize.ttest(differenceBtwAssessNTestPTTSpeedEXP,
                                                 differenceBtwAssessNTestToHSpeedEXP, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTSpeedEXP= mean(differenceBtwAssessNTestPTTSpeedEXP)
pstdevDifferenceBtwAssessNTestPTTSpeedEXP= pstdev(differenceBtwAssessNTestPTTSpeedEXP)

meanDifferenceBtwAssessNTestToHSpeedEXP= mean(differenceBtwAssessNTestToHSpeedEXP)
pstdevDifferenceBtwAssessNTestToHSpeedEXP= pstdev(differenceBtwAssessNTestToHSpeedEXP)

differenceTaskTypeComparisonSpeedEXP= meanDifferenceBtwAssessNTestPTTSpeedEXP- meanDifferenceBtwAssessNTestToHSpeedEXP
fractionalDifferenceTaskTypeComparisonSpeedEXP= differenceTaskTypeComparisonSpeedEXP/meanDifferenceBtwAssessNTestPTTSpeedEXP

TaskTypeComparisonSpeedEXP.insert(0, "Name", "Task Type Comparisons for Speed EXP")
TaskTypeComparisonSpeedEXP.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTSpeedEXP)
TaskTypeComparisonSpeedEXP.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTSpeedEXP)
TaskTypeComparisonSpeedEXP.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHSpeedEXP)
TaskTypeComparisonSpeedEXP.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHSpeedEXP)
TaskTypeComparisonSpeedEXP.insert(13, "Diff In Seconds", differenceTaskTypeComparisonSpeedEXP)
TaskTypeComparisonSpeedEXP.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonSpeedEXP)

space = [""]
TaskTypeComparisonsEXPforPerformanceMetricsHeader = pd.DataFrame(data={"": space})
TaskTypeComparisonsEXPforPerformanceMetricsHeader.insert(0, "Name", "Task Type Comparisons for Each Performance Metric in EXP")



# Stats of Task Type Comparisons Control Comparison
PTTDiffResumeCONAssess = ControlComparisonResumptionLagsAssessmentPathTracing
PTTDiffResumeCONTest = ControlComparisonResumptionLagsTestingPathTracing
differenceBtwAssessNTestPTTResumeCON = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffResumeCONAssess, PTTDiffResumeCONTest)]

ToHDiffResumeCONAssess = ControlComparisonResumptionLagsAssessmentToH
ToHDiffResumeCONTest = ControlComparisonResumptionLagsTestingToH
differenceBtwAssessNTestToHResumeCON = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffResumeCONAssess, ToHDiffResumeCONTest)]

# Task Type Comparisons for Resumption
TaskTypeComparisonResumeCON = statisticize.ttest(differenceBtwAssessNTestPTTResumeCON,
                                                 differenceBtwAssessNTestToHResumeCON, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTResumeCON= mean(differenceBtwAssessNTestPTTResumeCON)
pstdevDifferenceBtwAssessNTestPTTResumeCON= pstdev(differenceBtwAssessNTestPTTResumeCON)

meanDifferenceBtwAssessNTestToHResumeCON= mean(differenceBtwAssessNTestToHResumeCON)
pstdevDifferenceBtwAssessNTestToHResumeCON= pstdev(differenceBtwAssessNTestToHResumeCON)

differenceTaskTypeComparisonResumeCON= meanDifferenceBtwAssessNTestPTTResumeCON- meanDifferenceBtwAssessNTestToHResumeCON
fractionalDifferenceTaskTypeComparisonResumeCON= differenceTaskTypeComparisonResumeCON/meanDifferenceBtwAssessNTestPTTResumeCON

TaskTypeComparisonResumeCON.insert(0, "Name", "Task Type Comparisons for Resumption CON")
TaskTypeComparisonResumeCON.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTResumeCON)
TaskTypeComparisonResumeCON.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTResumeCON)
TaskTypeComparisonResumeCON.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHResumeCON)
TaskTypeComparisonResumeCON.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHResumeCON)
TaskTypeComparisonResumeCON.insert(13, "Diff In Seconds", differenceTaskTypeComparisonResumeCON)
TaskTypeComparisonResumeCON.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonResumeCON)


PTTDiffAttendCONAssess = ControlComparisonInterruptionLagsAssessmentPathTracing
PTTDiffAttendCONTest = ControlComparisonInterruptionLagsTestingPathTracing
differenceBtwAssessNTestPTTAttendCON = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffAttendCONAssess, PTTDiffAttendCONTest)]

ToHDiffAttendCONAssess = ControlComparisonInterruptionLagsAssessmentToH
ToHDiffAttendCONATest = ControlComparisonInterruptionLagsTestingToH
differenceBtwAssessNTestToHAttendCON = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffAttendCONAssess, ToHDiffAttendCONATest)]

# Task Type Comparisons for Interruption
TaskTypeComparisonAttendCON = statisticize.ttest(differenceBtwAssessNTestPTTAttendCON,
                                                 differenceBtwAssessNTestToHAttendCON, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTAttendCON= mean(differenceBtwAssessNTestPTTAttendCON)
pstdevDifferenceBtwAssessNTestPTTAttendCON= pstdev(differenceBtwAssessNTestPTTAttendCON)

meanDifferenceBtwAssessNTestToHAttendCON= mean(differenceBtwAssessNTestToHAttendCON)
pstdevDifferenceBtwAssessNTestToHAttendCON= pstdev(differenceBtwAssessNTestToHAttendCON)

differenceTaskTypeComparisonAttendCON= meanDifferenceBtwAssessNTestPTTAttendCON- meanDifferenceBtwAssessNTestToHAttendCON
fractionalDifferenceTaskTypeComparisonAttendCON= differenceTaskTypeComparisonAttendCON/meanDifferenceBtwAssessNTestPTTAttendCON

TaskTypeComparisonAttendCON.insert(0, "Name", "Task Type Comparisons for Interruption CON")
TaskTypeComparisonAttendCON.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTAttendCON)
TaskTypeComparisonAttendCON.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTAttendCON)
TaskTypeComparisonAttendCON.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHAttendCON)
TaskTypeComparisonAttendCON.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHAttendCON)
TaskTypeComparisonAttendCON.insert(13, "Diff In Seconds", differenceTaskTypeComparisonAttendCON)
TaskTypeComparisonAttendCON.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonAttendCON)


PTTDiffAccuracyCONAssess = ControlComparisonAccuraciesAssessmentPathTracing
PTTDiffAccuracyCONTest = ControlComparisonAccuraciesTestingPathTracing
differenceBtwAssessNTestPTTAccuracyCON = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffAccuracyCONAssess, PTTDiffAccuracyCONTest)]

ToHDiffAccuracyCONAssess = ControlComparisonAccuraciesAssessmentToH
ToHDiffAccuracyCONTest = ControlComparisonAccuraciesTestingToH
differenceBtwAssessNTestToHAccuracyCON = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffAccuracyCONAssess, ToHDiffAccuracyCONTest)]

# Task Type Comparisons for AccuracyCON
TaskTypeComparisonAccuracyCON = statisticize.ttest(differenceBtwAssessNTestPTTAccuracyCON,
                                                 differenceBtwAssessNTestToHAccuracyCON, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTAccuracyCON= mean(differenceBtwAssessNTestPTTAccuracyCON)
pstdevDifferenceBtwAssessNTestPTTAccuracyCON= pstdev(differenceBtwAssessNTestPTTAccuracyCON)

meanDifferenceBtwAssessNTestToHAccuracyCON= mean(differenceBtwAssessNTestToHAccuracyCON)
pstdevDifferenceBtwAssessNTestToHAccuracyCON= pstdev(differenceBtwAssessNTestToHAccuracyCON)

differenceTaskTypeComparisonAccuracyCON= meanDifferenceBtwAssessNTestPTTAccuracyCON- meanDifferenceBtwAssessNTestToHAccuracyCON
fractionalDifferenceTaskTypeComparisonAccuracyCON= differenceTaskTypeComparisonAccuracyCON

TaskTypeComparisonAccuracyCON.insert(0, "Name", "Task Type Comparisons for Accuracy CON")
TaskTypeComparisonAccuracyCON.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTAccuracyCON)
TaskTypeComparisonAccuracyCON.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTAccuracyCON)
TaskTypeComparisonAccuracyCON.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHAccuracyCON)
TaskTypeComparisonAccuracyCON.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHAccuracyCON)
TaskTypeComparisonAccuracyCON.insert(13, "Diff In Seconds", differenceTaskTypeComparisonAccuracyCON)
TaskTypeComparisonAccuracyCON.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonAccuracyCON)


PTTDiffSpeedCONAssess = ControlComparisonSpeedAssessmentPathTracing
PTTDiffSpeedCONTest = ControlComparisonSpeedTestingPathTracing
differenceBtwAssessNTestPTTSpeedCON = [sumAssess - sumTest for (sumAssess, sumTest) in zip(PTTDiffSpeedCONAssess, PTTDiffSpeedCONTest)]

ToHDiffSpeedCONAssess = ControlComparisonSpeedAssessmentToH
ToHDiffSpeedCONTest = ControlComparisonSpeedTestingToH
differenceBtwAssessNTestToHSpeedCON = [sumAssess - sumTest for (sumAssess, sumTest) in zip(ToHDiffSpeedCONAssess, ToHDiffSpeedCONTest)]

# Task Type Comparisons for Speed
TaskTypeComparisonSpeedCON = statisticize.ttest(differenceBtwAssessNTestPTTSpeedCON,
                                                 differenceBtwAssessNTestToHSpeedCON, paired=False)#, alternative="greater")

meanDifferenceBtwAssessNTestPTTSpeedCON= mean(differenceBtwAssessNTestPTTSpeedCON)
pstdevDifferenceBtwAssessNTestPTTSpeedCON= pstdev(differenceBtwAssessNTestPTTSpeedCON)

meanDifferenceBtwAssessNTestToHSpeedCON= mean(differenceBtwAssessNTestToHSpeedCON)
pstdevDifferenceBtwAssessNTestToHSpeedCON= pstdev(differenceBtwAssessNTestToHSpeedCON)

differenceTaskTypeComparisonSpeedCON= meanDifferenceBtwAssessNTestPTTSpeedCON- meanDifferenceBtwAssessNTestToHSpeedCON
fractionalDifferenceTaskTypeComparisonSpeedCON= differenceTaskTypeComparisonSpeedCON/meanDifferenceBtwAssessNTestPTTSpeedCON

TaskTypeComparisonSpeedCON.insert(0, "Name", "Task Type Comparisons for Speed CON")
TaskTypeComparisonSpeedCON.insert(9, "Mean Sample 1 (Seconds)", meanDifferenceBtwAssessNTestPTTSpeedCON)
TaskTypeComparisonSpeedCON.insert(10, "SD Sample 1 (Seconds)", pstdevDifferenceBtwAssessNTestPTTSpeedCON)
TaskTypeComparisonSpeedCON.insert(11, "Mean Sample 2 (Seconds)", meanDifferenceBtwAssessNTestToHSpeedCON)
TaskTypeComparisonSpeedCON.insert(12, "SD Sample 2 (Seconds)", pstdevDifferenceBtwAssessNTestToHSpeedCON)
TaskTypeComparisonSpeedCON.insert(13, "Diff In Seconds", differenceTaskTypeComparisonSpeedCON)
TaskTypeComparisonSpeedCON.insert(14, "Fraction In Seconds", fractionalDifferenceTaskTypeComparisonSpeedCON)

space = [""]
TaskTypeComparisonsCONforPerformanceMetricsHeader = pd.DataFrame(data={"": space})
TaskTypeComparisonsCONforPerformanceMetricsHeader.insert(0, "Name", "Task Type Comparisons for Each Performance Metric in CON")

# Stats Sorted and Grouped by Primary Task and H1
# All 60 datapoints per metric per participant per hypothesis within the Experimental Intervention

ExperimentalInterventionResumptionLagsAssessmentPathTracingH1=\
ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment +\
ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment

ExperimentalInterventionResumptionLagsAssessmentToHH1=\
ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment

ExperimentalInterventionResumptionLagsTestingPathTracingH1=\
ExpH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting+\
ExpH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting

ExperimentalInterventionResumptionLagsTestingToHH1=\
ExpH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting

ExperimentalInterventionInterruptionLagsAssessmentPathTracingH1=\
ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment+\
ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment

ExperimentalInterventionInterruptionLagsAssessmentToHH1=\
ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment

ExperimentalInterventionInterruptionLagsTestingPathTracingH1=\
ExpH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting+\
ExpH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting

ExperimentalInterventionInterruptionLagsTestingToHH1=\
ExpH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting

ExperimentalInterventionAccuraciesAssessmentPathTracingH1=\
ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment+\
ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment

ExperimentalInterventionAccuraciesAssessmentToHH1=\
ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment

ExperimentalInterventionAccuraciesTestingPathTracingH1=\
ExpH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting+\
ExpH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting

ExperimentalInterventionAccuraciesTestingToHH1=\
ExpH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting+\
ExpH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting

ExperimentalInterventionSpeedAssessmentPathTracingH1= \
ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment +\
ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment

ExperimentalInterventionSpeedAssessmentToHH1= \
ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment+\
ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment

ExperimentalInterventionSpeedTestingPathTracingH1= \
ExpH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting + \
ExpH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting

ExperimentalInterventionSpeedTestingToHH1= \
ExpH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting + \
ExpH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting


# All 60 datapoints per metric per participant per hypothesis within the Control Comparison
ControlComparisonResumptionLagsAssessmentPathTracingH1=\
ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsAssessment +\
ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsAssessment

ControlComparisonResumptionLagsAssessmentToHH1=\
ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsAssessment

ControlComparisonResumptionLagsTestingPathTracingH1=\
ControlH1DrawHanoiDrawStroopCollectedSumResumptionLagsTesting+\
ControlH1DrawHanoiDrawMathCollectedSumResumptionLagsTesting

ControlComparisonResumptionLagsTestingToHH1=\
ControlH1HanoiDrawHanoiStroopCollectedSumResumptionLagsTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumResumptionLagsTesting

ControlComparisonInterruptionLagsAssessmentPathTracingH1=\
ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsAssessment+\
ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsAssessment

ControlComparisonInterruptionLagsAssessmentToHH1=\
ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsAssessment

ControlComparisonInterruptionLagsTestingPathTracingH1=\
ControlH1DrawHanoiDrawStroopCollectedSumInterruptionLagsTesting+\
ControlH1DrawHanoiDrawMathCollectedSumInterruptionLagsTesting

ControlComparisonInterruptionLagsTestingToHH1=\
ControlH1HanoiDrawHanoiStroopCollectedSumInterruptionLagsTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumInterruptionLagsTesting

ControlComparisonAccuraciesAssessmentPathTracingH1=\
ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesAssessment+\
ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesAssessment

ControlComparisonAccuraciesAssessmentToHH1=\
ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesAssessment

ControlComparisonAccuraciesTestingPathTracingH1=\
ControlH1DrawHanoiDrawStroopCollectedSumsMovesAndSequencesTesting+\
ControlH1DrawHanoiDrawMathCollectedSumsMovesAndSequencesTesting

ControlComparisonAccuraciesTestingToHH1=\
ControlH1HanoiDrawHanoiStroopCollectedSumsMovesAndSequencesTesting+\
ControlH1HanoiDrawHanoiMathCollectedSumsMovesAndSequencesTesting

ControlComparisonSpeedAssessmentPathTracingH1= \
ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesAssessment +\
ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesAssessment

ControlComparisonSpeedAssessmentToHH1= \
ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesAssessment+\
ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesAssessment

ControlComparisonSpeedTestingPathTracingH1= \
ControlH1DrawHanoiDrawStroopCollectedSumsCompletionTimesTesting + \
ControlH1DrawHanoiDrawMathCollectedSumsCompletionTimesTesting

ControlComparisonSpeedTestingToHH1= \
ControlH1HanoiDrawHanoiStroopCollectedSumsCompletionTimesTesting + \
ControlH1HanoiDrawHanoiMathCollectedSumsCompletionTimesTesting


space = [""]
AssessTestingPathTracingH1ExperimentalHeader = pd.DataFrame(data={"": space})
AssessTestingPathTracingH1ExperimentalHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for Path-Tracing in H1 of Experimental Intervention")

space = [""]
AssessTestingInToHH1ExperimentalHeader = pd.DataFrame(data={"": space})
AssessTestingInToHH1ExperimentalHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for ToH in H1 of Experimental Intervention")

space = [""]
AssessTestingPathTracingH1ControlHeader = pd.DataFrame(data={"": space})
AssessTestingPathTracingH1ControlHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for Path-Tracing in H1 of Control Comparison")

space = [""]
AssessTestingToHH1ControlHeader = pd.DataFrame(data={"": space})
AssessTestingToHH1ControlHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for ToH in H1 of Control Comparison")


# Path Tracing of Experimental Intervention Resumption Stats
PathTracingExperimentalInterventionResumptionLagsH1 = statisticize.ttest(ExperimentalInterventionResumptionLagsAssessmentPathTracingH1,
                                                 ExperimentalInterventionResumptionLagsTestingPathTracingH1, paired=True, alternative="greater")

meanExperimentalInterventionResumptionLagsAssessmentPathTracingH1 = mean(ExperimentalInterventionResumptionLagsAssessmentPathTracingH1)
pstdevExperimentalInterventionResumptionLagsAssessmentPathTracingH1 = pstdev(ExperimentalInterventionResumptionLagsAssessmentPathTracingH1)

meanExperimentalInterventionResumptionLagsTestingPathTracingH1 = mean(ExperimentalInterventionResumptionLagsTestingPathTracingH1)
pstdevExperimentalInterventionResumptionLagsTestingPathTracingH1 = pstdev(ExperimentalInterventionResumptionLagsTestingPathTracingH1)

differenceBtwAssessResumePathTracingH1 = meanExperimentalInterventionResumptionLagsAssessmentPathTracingH1 - meanExperimentalInterventionResumptionLagsTestingPathTracingH1
fractionalDifferenceBtwAssessResumePathTracingH1 = differenceBtwAssessResumePathTracingH1/meanExperimentalInterventionResumptionLagsAssessmentPathTracingH1

PathTracingExperimentalInterventionResumptionLagsH1.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Path Tracing of Experimental Intervention H1")
PathTracingExperimentalInterventionResumptionLagsH1.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionResumptionLagsAssessmentPathTracingH1)
PathTracingExperimentalInterventionResumptionLagsH1.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionResumptionLagsAssessmentPathTracingH1)
PathTracingExperimentalInterventionResumptionLagsH1.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionResumptionLagsTestingPathTracingH1)
PathTracingExperimentalInterventionResumptionLagsH1.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionResumptionLagsTestingPathTracingH1)
PathTracingExperimentalInterventionResumptionLagsH1.insert(13, "Diff In Seconds", differenceBtwAssessResumePathTracingH1)
PathTracingExperimentalInterventionResumptionLagsH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumePathTracingH1)


# ToH of Experimental Intervention Resumption Stats
ToHExperimentalInterventionResumptionLagsH1 = statisticize.ttest(ExperimentalInterventionResumptionLagsAssessmentToHH1,
                                                 ExperimentalInterventionResumptionLagsTestingToHH1, paired=True, alternative="greater")

meanExperimentalInterventionResumptionLagsAssessmentToHH1= mean(ExperimentalInterventionResumptionLagsAssessmentToHH1)
pstdevExperimentalInterventionResumptionLagsAssessmentToHH1= pstdev(ExperimentalInterventionResumptionLagsAssessmentToHH1)

meanExperimentalInterventionResumptionLagsTestingToHH1= mean(ExperimentalInterventionResumptionLagsTestingToHH1)
pstdevExperimentalInterventionResumptionLagsTestingToHH1= pstdev(ExperimentalInterventionResumptionLagsTestingToHH1)

differenceBtwAssessResumeToHH1= meanExperimentalInterventionResumptionLagsAssessmentToHH1- meanExperimentalInterventionResumptionLagsTestingToHH1
fractionalDifferenceBtwAssessResumeToHH1= differenceBtwAssessResumeToHH1/meanExperimentalInterventionResumptionLagsAssessmentToHH1

ToHExperimentalInterventionResumptionLagsH1.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing ToH in H1 of Experimental Intervention")
ToHExperimentalInterventionResumptionLagsH1.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionResumptionLagsAssessmentToHH1)
ToHExperimentalInterventionResumptionLagsH1.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionResumptionLagsAssessmentToHH1)
ToHExperimentalInterventionResumptionLagsH1.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionResumptionLagsTestingToHH1)
ToHExperimentalInterventionResumptionLagsH1.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionResumptionLagsTestingToHH1)
ToHExperimentalInterventionResumptionLagsH1.insert(13, "Diff In Seconds", differenceBtwAssessResumeToHH1)
ToHExperimentalInterventionResumptionLagsH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeToHH1)


# Path Tracing of Experimental Intervention Interruption Stats
PathTracingExperimentalInterventionInterruptionLagsH1 = statisticize.ttest(ExperimentalInterventionInterruptionLagsAssessmentPathTracingH1,
                                                 ExperimentalInterventionInterruptionLagsTestingPathTracingH1, paired=True, alternative="greater")

meanExperimentalInterventionInterruptionLagsAssessmentPathTracingH1 = mean(ExperimentalInterventionInterruptionLagsAssessmentPathTracingH1)
pstdevExperimentalInterventionInterruptionLagsAssessmentPathTracingH1 = pstdev(ExperimentalInterventionInterruptionLagsAssessmentPathTracingH1)

meanExperimentalInterventionInterruptionLagsTestingPathTracingH1 = mean(ExperimentalInterventionInterruptionLagsTestingPathTracingH1)
pstdevExperimentalInterventionInterruptionLagsTestingPathTracingH1 = pstdev(ExperimentalInterventionInterruptionLagsTestingPathTracingH1)

differenceBtwAssessAttendPathTracingH1 = meanExperimentalInterventionInterruptionLagsAssessmentPathTracingH1 - meanExperimentalInterventionInterruptionLagsTestingPathTracingH1
fractionalDifferenceBtwAssessAttendPathTracingH1 = differenceBtwAssessAttendPathTracingH1/meanExperimentalInterventionInterruptionLagsAssessmentPathTracingH1

PathTracingExperimentalInterventionInterruptionLagsH1.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Path Tracing in H1 of Experimental Intervention")
PathTracingExperimentalInterventionInterruptionLagsH1.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionInterruptionLagsAssessmentPathTracingH1)
PathTracingExperimentalInterventionInterruptionLagsH1.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionInterruptionLagsAssessmentPathTracingH1)
PathTracingExperimentalInterventionInterruptionLagsH1.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionInterruptionLagsTestingPathTracingH1)
PathTracingExperimentalInterventionInterruptionLagsH1.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionInterruptionLagsTestingPathTracingH1)
PathTracingExperimentalInterventionInterruptionLagsH1.insert(13, "Diff In Seconds", differenceBtwAssessAttendPathTracingH1)
PathTracingExperimentalInterventionInterruptionLagsH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendPathTracingH1)


# ToH of Experimental Intervention Interruption Stats
ToHExperimentalInterventionInterruptionLagsH1 = statisticize.ttest(ExperimentalInterventionInterruptionLagsAssessmentToHH1,
                                                 ExperimentalInterventionInterruptionLagsTestingToHH1, paired=True, alternative="greater")

meanExperimentalInterventionInterruptionLagsAssessmentToHH1= mean(ExperimentalInterventionInterruptionLagsAssessmentToHH1)
pstdevExperimentalInterventionInterruptionLagsAssessmentToHH1= pstdev(ExperimentalInterventionInterruptionLagsAssessmentToHH1)

meanExperimentalInterventionInterruptionLagsTestingToHH1= mean(ExperimentalInterventionInterruptionLagsTestingToHH1)
pstdevExperimentalInterventionInterruptionLagsTestingToHH1= pstdev(ExperimentalInterventionInterruptionLagsTestingToHH1)

differenceBtwAssessAttendToHH1= meanExperimentalInterventionInterruptionLagsAssessmentToHH1- meanExperimentalInterventionInterruptionLagsTestingToHH1
fractionalDifferenceBtwAssessAttendToHH1= differenceBtwAssessAttendToHH1/meanExperimentalInterventionInterruptionLagsAssessmentToHH1

ToHExperimentalInterventionInterruptionLagsH1.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing ToH in H1 of Experimental Intervention")
ToHExperimentalInterventionInterruptionLagsH1.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionInterruptionLagsAssessmentToHH1)
ToHExperimentalInterventionInterruptionLagsH1.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionInterruptionLagsAssessmentToHH1)
ToHExperimentalInterventionInterruptionLagsH1.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionInterruptionLagsTestingToHH1)
ToHExperimentalInterventionInterruptionLagsH1.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionInterruptionLagsTestingToHH1)
ToHExperimentalInterventionInterruptionLagsH1.insert(13, "Diff In Seconds", differenceBtwAssessAttendToHH1)
ToHExperimentalInterventionInterruptionLagsH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendToHH1)


# Path Tracing of Experimental Intervention Accuracy Stats
PathTracingExperimentalInterventionAccuraciesH1 = statisticize.ttest(ExperimentalInterventionAccuraciesAssessmentPathTracingH1,
                                                 ExperimentalInterventionAccuraciesTestingPathTracingH1, paired=True, alternative="greater")

meanExperimentalInterventionAccuraciesAssessmentPathTracingH1 = mean(ExperimentalInterventionAccuraciesAssessmentPathTracingH1)
pstdevExperimentalInterventionAccuraciesAssessmentPathTracingH1 = pstdev(ExperimentalInterventionAccuraciesAssessmentPathTracingH1)

meanExperimentalInterventionAccuraciesTestingPathTracingH1 = mean(ExperimentalInterventionAccuraciesTestingPathTracingH1)
pstdevExperimentalInterventionAccuraciesTestingPathTracingH1 = pstdev(ExperimentalInterventionAccuraciesTestingPathTracingH1)

differenceBtwAssessAccuraciesPathTracingH1 = meanExperimentalInterventionAccuraciesAssessmentPathTracingH1 - meanExperimentalInterventionAccuraciesTestingPathTracingH1
fractionalDifferenceBtwAssessAccuraciesPathTracingH1 = differenceBtwAssessAccuraciesPathTracingH1

PathTracingExperimentalInterventionAccuraciesH1.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Path Tracing in H1 of Experimental Intervention")
PathTracingExperimentalInterventionAccuraciesH1.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionAccuraciesAssessmentPathTracingH1)
PathTracingExperimentalInterventionAccuraciesH1.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionAccuraciesAssessmentPathTracingH1)
PathTracingExperimentalInterventionAccuraciesH1.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionAccuraciesTestingPathTracingH1)
PathTracingExperimentalInterventionAccuraciesH1.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionAccuraciesTestingPathTracingH1)
PathTracingExperimentalInterventionAccuraciesH1.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesPathTracingH1)
PathTracingExperimentalInterventionAccuraciesH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesPathTracingH1)


# ToH of Experimental Intervention Accuracy Stats
ToHExperimentalInterventionAccuraciesH1 = statisticize.ttest(ExperimentalInterventionAccuraciesAssessmentToHH1,
                                                 ExperimentalInterventionAccuraciesTestingToHH1, paired=True, alternative="greater")

meanExperimentalInterventionAccuraciesAssessmentToHH1= mean(ExperimentalInterventionAccuraciesAssessmentToHH1)
pstdevExperimentalInterventionAccuraciesAssessmentToHH1= pstdev(ExperimentalInterventionAccuraciesAssessmentToHH1)

meanExperimentalInterventionAccuraciesTestingToHH1= mean(ExperimentalInterventionAccuraciesTestingToHH1)
pstdevExperimentalInterventionAccuraciesTestingToHH1= pstdev(ExperimentalInterventionAccuraciesTestingToHH1)

differenceBtwAssessAccuraciesToHH1= meanExperimentalInterventionAccuraciesAssessmentToHH1- meanExperimentalInterventionAccuraciesTestingToHH1
fractionalDifferenceBtwAssessAccuraciesToHH1= differenceBtwAssessAccuraciesToHH1

ToHExperimentalInterventionAccuraciesH1.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing ToH in H1 of Experimental Intervention")
ToHExperimentalInterventionAccuraciesH1.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionAccuraciesAssessmentToHH1)
ToHExperimentalInterventionAccuraciesH1.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionAccuraciesAssessmentToHH1)
ToHExperimentalInterventionAccuraciesH1.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionAccuraciesTestingToHH1)
ToHExperimentalInterventionAccuraciesH1.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionAccuraciesTestingToHH1)
ToHExperimentalInterventionAccuraciesH1.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesToHH1)
ToHExperimentalInterventionAccuraciesH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesToHH1)


# Path Tracing of Experimental Intervention Speed Stats
PathTracingExperimentalInterventionSpeedH1 = statisticize.ttest(ExperimentalInterventionSpeedAssessmentPathTracingH1,
                                                 ExperimentalInterventionSpeedTestingPathTracingH1, paired=True, alternative="greater")

meanExperimentalInterventionSpeedAssessmentPathTracingH1 = mean(ExperimentalInterventionSpeedAssessmentPathTracingH1)
pstdevExperimentalInterventionSpeedAssessmentPathTracingH1 = pstdev(ExperimentalInterventionSpeedAssessmentPathTracingH1)

meanExperimentalInterventionSpeedTestingPathTracingH1 = mean(ExperimentalInterventionSpeedTestingPathTracingH1)
pstdevExperimentalInterventionSpeedTestingPathTracingH1 = pstdev(ExperimentalInterventionSpeedTestingPathTracingH1)

differenceBtwAssessSpeedsPathTracingH1 = meanExperimentalInterventionSpeedAssessmentPathTracingH1 - meanExperimentalInterventionSpeedTestingPathTracingH1
fractionalDifferenceBtwAssessSpeedsPathTracingH1 = differenceBtwAssessSpeedsPathTracingH1/meanExperimentalInterventionSpeedAssessmentPathTracingH1

PathTracingExperimentalInterventionSpeedH1.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Path Tracing in H1 of Experimental Intervention")
PathTracingExperimentalInterventionSpeedH1.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionSpeedAssessmentPathTracingH1)
PathTracingExperimentalInterventionSpeedH1.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionSpeedAssessmentPathTracingH1)
PathTracingExperimentalInterventionSpeedH1.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionSpeedTestingPathTracingH1)
PathTracingExperimentalInterventionSpeedH1.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionSpeedTestingPathTracingH1)
PathTracingExperimentalInterventionSpeedH1.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsPathTracingH1)
PathTracingExperimentalInterventionSpeedH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsPathTracingH1)

# ToH of Experimental Intervention Speed Stats
ToHExperimentalInterventionSpeedH1 = statisticize.ttest(ExperimentalInterventionSpeedAssessmentToHH1,
                                                 ExperimentalInterventionSpeedTestingToHH1, paired=True, alternative="greater")

meanExperimentalInterventionSpeedAssessmentToHH1= mean(ExperimentalInterventionSpeedAssessmentToHH1)
pstdevExperimentalInterventionSpeedAssessmentToHH1= pstdev(ExperimentalInterventionSpeedAssessmentToHH1)

meanExperimentalInterventionSpeedTestingToHH1= mean(ExperimentalInterventionSpeedTestingToHH1)
pstdevExperimentalInterventionSpeedTestingToHH1= pstdev(ExperimentalInterventionSpeedTestingToHH1)

differenceBtwAssessSpeedsToHH1= meanExperimentalInterventionSpeedAssessmentToHH1- meanExperimentalInterventionSpeedTestingToHH1
fractionalDifferenceBtwAssessSpeedsToHH1= differenceBtwAssessSpeedsToHH1/meanExperimentalInterventionSpeedAssessmentToHH1

ToHExperimentalInterventionSpeedH1.insert(0, "Name", "Speed's Stats BTW Assessment and Testing ToH in H1 of Experimental Intervention")
ToHExperimentalInterventionSpeedH1.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionSpeedAssessmentToHH1)
ToHExperimentalInterventionSpeedH1.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionSpeedAssessmentToHH1)
ToHExperimentalInterventionSpeedH1.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionSpeedTestingToHH1)
ToHExperimentalInterventionSpeedH1.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionSpeedTestingToHH1)
ToHExperimentalInterventionSpeedH1.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsToHH1)
ToHExperimentalInterventionSpeedH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsToHH1)



# Control Comparison Stats
# Path Tracing of Control Comparison Resumption Stats
PathTracingControlComparisonResumptionLagsH1 = statisticize.ttest(ControlComparisonResumptionLagsAssessmentPathTracingH1,
                                                 ControlComparisonResumptionLagsTestingPathTracingH1, paired=True, alternative="greater")

meanControlComparisonResumptionLagsAssessmentPathTracingH1 = mean(ControlComparisonResumptionLagsAssessmentPathTracingH1)
pstdevControlComparisonResumptionLagsAssessmentPathTracingH1 = pstdev(ControlComparisonResumptionLagsAssessmentPathTracingH1)

meanControlComparisonResumptionLagsTestingPathTracingH1 = mean(ControlComparisonResumptionLagsTestingPathTracingH1)
pstdevControlComparisonResumptionLagsTestingPathTracingH1 = pstdev(ControlComparisonResumptionLagsTestingPathTracingH1)

differenceBtwAssessResumeControlPathTracingH1 = meanControlComparisonResumptionLagsAssessmentPathTracingH1 - meanControlComparisonResumptionLagsTestingPathTracingH1
fractionalDifferenceBtwAssessResumeControlPathTracingH1 = differenceBtwAssessResumeControlPathTracingH1/meanControlComparisonResumptionLagsAssessmentPathTracingH1

PathTracingControlComparisonResumptionLagsH1.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Path Tracing in H1 of Control Comparison")
PathTracingControlComparisonResumptionLagsH1.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonResumptionLagsAssessmentPathTracingH1)
PathTracingControlComparisonResumptionLagsH1.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonResumptionLagsAssessmentPathTracingH1)
PathTracingControlComparisonResumptionLagsH1.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonResumptionLagsTestingPathTracingH1)
PathTracingControlComparisonResumptionLagsH1.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonResumptionLagsTestingPathTracingH1)
PathTracingControlComparisonResumptionLagsH1.insert(13, "Diff In Seconds", differenceBtwAssessResumeControlPathTracingH1)
PathTracingControlComparisonResumptionLagsH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeControlPathTracingH1)


# To Hof Control Comparison Resumption Stats
ToHControlComparisonResumptionLagsH1 = statisticize.ttest(ControlComparisonResumptionLagsAssessmentToHH1,
                                                 ControlComparisonResumptionLagsTestingToHH1, paired=True, alternative="greater")

meanControlComparisonResumptionLagsAssessmentToHH1= mean(ControlComparisonResumptionLagsAssessmentToHH1)
pstdevControlComparisonResumptionLagsAssessmentToHH1= pstdev(ControlComparisonResumptionLagsAssessmentToHH1)

meanControlComparisonResumptionLagsTestingToHH1= mean(ControlComparisonResumptionLagsTestingToHH1)
pstdevControlComparisonResumptionLagsTestingToHH1= pstdev(ControlComparisonResumptionLagsTestingToHH1)

differenceBtwAssessResumeControlToHH1= meanControlComparisonResumptionLagsAssessmentToHH1- meanControlComparisonResumptionLagsTestingToHH1
fractionalDifferenceBtwAssessResumeControlToHH1= differenceBtwAssessResumeControlToHH1/meanControlComparisonResumptionLagsAssessmentToHH1

ToHControlComparisonResumptionLagsH1.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing ToH in H1 of Control Comparison")
ToHControlComparisonResumptionLagsH1.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonResumptionLagsAssessmentToHH1)
ToHControlComparisonResumptionLagsH1.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonResumptionLagsAssessmentToHH1)
ToHControlComparisonResumptionLagsH1.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonResumptionLagsTestingToHH1)
ToHControlComparisonResumptionLagsH1.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonResumptionLagsTestingToHH1)
ToHControlComparisonResumptionLagsH1.insert(13, "Diff In Seconds", differenceBtwAssessResumeControlToHH1)
ToHControlComparisonResumptionLagsH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeControlToHH1)


# Path Tracing of Control Comparison Interruption Stats
PathTracingControlComparisonInterruptionLagsH1 = statisticize.ttest(ControlComparisonInterruptionLagsAssessmentPathTracingH1,
                                                 ControlComparisonInterruptionLagsTestingPathTracingH1, paired=True, alternative="greater")

meanControlComparisonInterruptionLagsAssessmentPathTracingH1 = mean(ControlComparisonInterruptionLagsAssessmentPathTracingH1)
pstdevControlComparisonInterruptionLagsAssessmentPathTracingH1 = pstdev(ControlComparisonInterruptionLagsAssessmentPathTracingH1)

meanControlComparisonInterruptionLagsTestingPathTracingH1 = mean(ControlComparisonInterruptionLagsTestingPathTracingH1)
pstdevControlComparisonInterruptionLagsTestingPathTracingH1 = pstdev(ControlComparisonInterruptionLagsTestingPathTracingH1)

differenceBtwAssessAttendControlPathTracingH1 = meanControlComparisonInterruptionLagsAssessmentPathTracingH1 - meanControlComparisonInterruptionLagsTestingPathTracingH1
fractionalDifferenceBtwAssessAttendControlPathTracingH1 = differenceBtwAssessAttendControlPathTracingH1/meanControlComparisonInterruptionLagsAssessmentPathTracingH1

PathTracingControlComparisonInterruptionLagsH1.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Path Tracing in H1 of Control Comparison")
PathTracingControlComparisonInterruptionLagsH1.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonInterruptionLagsAssessmentPathTracingH1)
PathTracingControlComparisonInterruptionLagsH1.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonInterruptionLagsAssessmentPathTracingH1)
PathTracingControlComparisonInterruptionLagsH1.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonInterruptionLagsTestingPathTracingH1)
PathTracingControlComparisonInterruptionLagsH1.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonInterruptionLagsTestingPathTracingH1)
PathTracingControlComparisonInterruptionLagsH1.insert(13, "Diff In Seconds", differenceBtwAssessAttendControlPathTracingH1)
PathTracingControlComparisonInterruptionLagsH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendControlPathTracingH1)


# ToH of Control Comparison Interruption Stats
ToHControlComparisonInterruptionLagsH1 = statisticize.ttest(ControlComparisonInterruptionLagsAssessmentToHH1,
                                                 ControlComparisonInterruptionLagsTestingToHH1, paired=True, alternative="greater")

meanControlComparisonInterruptionLagsAssessmentToHH1= mean(ControlComparisonInterruptionLagsAssessmentToHH1)
pstdevControlComparisonInterruptionLagsAssessmentToHH1= pstdev(ControlComparisonInterruptionLagsAssessmentToHH1)

meanControlComparisonInterruptionLagsTestingToHH1= mean(ControlComparisonInterruptionLagsTestingToHH1)
pstdevControlComparisonInterruptionLagsTestingToHH1= pstdev(ControlComparisonInterruptionLagsTestingToHH1)

differenceBtwAssessAttendControlToHH1= meanControlComparisonInterruptionLagsAssessmentToHH1- meanControlComparisonInterruptionLagsTestingToHH1
fractionalDifferenceBtwAssessAttendControlToHH1= differenceBtwAssessAttendControlToHH1/meanControlComparisonInterruptionLagsAssessmentToHH1

ToHControlComparisonInterruptionLagsH1.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing ToH in H1 of Control Comparison")
ToHControlComparisonInterruptionLagsH1.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonInterruptionLagsAssessmentToHH1)
ToHControlComparisonInterruptionLagsH1.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonInterruptionLagsAssessmentToHH1)
ToHControlComparisonInterruptionLagsH1.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonInterruptionLagsTestingToHH1)
ToHControlComparisonInterruptionLagsH1.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonInterruptionLagsTestingToHH1)
ToHControlComparisonInterruptionLagsH1.insert(13, "Diff In Seconds", differenceBtwAssessAttendControlToHH1)
ToHControlComparisonInterruptionLagsH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendControlToHH1)


# Path Tracing of Control Comparison Accuracy Stats
PathTracingControlComparisonAccuraciesH1 = statisticize.ttest(ControlComparisonAccuraciesAssessmentPathTracingH1,
                                                 ControlComparisonAccuraciesTestingPathTracingH1, paired=True, alternative="greater")

meanControlComparisonAccuraciesAssessmentPathTracingH1 = mean(ControlComparisonAccuraciesAssessmentPathTracingH1)
pstdevControlComparisonAccuraciesAssessmentPathTracingH1 = pstdev(ControlComparisonAccuraciesAssessmentPathTracingH1)

meanControlComparisonAccuraciesTestingPathTracingH1 = mean(ControlComparisonAccuraciesTestingPathTracingH1)
pstdevControlComparisonAccuraciesTestingPathTracingH1 = pstdev(ControlComparisonAccuraciesTestingPathTracingH1)

differenceBtwAssessAccuraciesControlPathTracingH1 = meanControlComparisonAccuraciesAssessmentPathTracingH1 - meanControlComparisonAccuraciesTestingPathTracingH1
fractionalDifferenceBtwAssessAccuraciesControlPathTracingH1 = differenceBtwAssessAccuraciesControlPathTracingH1

PathTracingControlComparisonAccuraciesH1.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Path Tracing in H1 of Control Comparison")
PathTracingControlComparisonAccuraciesH1.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonAccuraciesAssessmentPathTracingH1)
PathTracingControlComparisonAccuraciesH1.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonAccuraciesAssessmentPathTracingH1)
PathTracingControlComparisonAccuraciesH1.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonAccuraciesTestingPathTracingH1)
PathTracingControlComparisonAccuraciesH1.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonAccuraciesTestingPathTracingH1)
PathTracingControlComparisonAccuraciesH1.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesControlPathTracingH1)
PathTracingControlComparisonAccuraciesH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesControlPathTracingH1)


# ToH of Control Comparison Accuracy Stats
ToHControlComparisonAccuraciesH1 = statisticize.ttest(ControlComparisonAccuraciesAssessmentToHH1,
                                                 ControlComparisonAccuraciesTestingToHH1, paired=True, alternative="greater")

meanControlComparisonAccuraciesAssessmentToHH1= mean(ControlComparisonAccuraciesAssessmentToHH1)
pstdevControlComparisonAccuraciesAssessmentToHH1= pstdev(ControlComparisonAccuraciesAssessmentToHH1)

meanControlComparisonAccuraciesTestingToHH1= mean(ControlComparisonAccuraciesTestingToHH1)
pstdevControlComparisonAccuraciesTestingToHH1= pstdev(ControlComparisonAccuraciesTestingToHH1)

differenceBtwAssessAccuraciesControlToHH1= meanControlComparisonAccuraciesAssessmentToHH1- meanControlComparisonAccuraciesTestingToHH1
fractionalDifferenceBtwAssessAccuraciesControlToHH1= differenceBtwAssessAccuraciesControlToHH1

ToHControlComparisonAccuraciesH1.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing ToH in H1 of Control Comparison")
ToHControlComparisonAccuraciesH1.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonAccuraciesAssessmentToHH1)
ToHControlComparisonAccuraciesH1.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonAccuraciesAssessmentToHH1)
ToHControlComparisonAccuraciesH1.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonAccuraciesTestingToHH1)
ToHControlComparisonAccuraciesH1.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonAccuraciesTestingToHH1)
ToHControlComparisonAccuraciesH1.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesControlToHH1)
ToHControlComparisonAccuraciesH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesControlToHH1)


# Path Tracing of Control Comparison Speed Stats
PathTracingControlComparisonSpeedH1 = statisticize.ttest(ControlComparisonSpeedAssessmentPathTracingH1,
                                                 ControlComparisonSpeedTestingPathTracingH1, paired=True, alternative="greater")

meanControlComparisonSpeedAssessmentPathTracingH1 = mean(ControlComparisonSpeedAssessmentPathTracingH1)
pstdevControlComparisonSpeedAssessmentPathTracingH1 = pstdev(ControlComparisonSpeedAssessmentPathTracingH1)

meanControlComparisonSpeedTestingPathTracingH1 = mean(ControlComparisonSpeedTestingPathTracingH1)
pstdevControlComparisonSpeedTestingPathTracingH1 = pstdev(ControlComparisonSpeedTestingPathTracingH1)

differenceBtwAssessSpeedsControlPathTracingH1 = meanControlComparisonSpeedAssessmentPathTracingH1 - meanControlComparisonSpeedTestingPathTracingH1
fractionalDifferenceBtwAssessSpeedsControlPathTracingH1 = differenceBtwAssessSpeedsControlPathTracingH1/meanControlComparisonSpeedAssessmentPathTracingH1

PathTracingControlComparisonSpeedH1.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Path Tracing in H1 of Control Comparison")
PathTracingControlComparisonSpeedH1.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonSpeedAssessmentPathTracingH1)
PathTracingControlComparisonSpeedH1.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonSpeedAssessmentPathTracingH1)
PathTracingControlComparisonSpeedH1.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonSpeedTestingPathTracingH1)
PathTracingControlComparisonSpeedH1.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonSpeedTestingPathTracingH1)
PathTracingControlComparisonSpeedH1.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsControlPathTracingH1)
PathTracingControlComparisonSpeedH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsControlPathTracingH1)


# ToH of Control Comparison Speed Stats
ToHControlComparisonSpeedH1 = statisticize.ttest(ControlComparisonSpeedAssessmentToHH1,
                                                 ControlComparisonSpeedTestingToHH1, paired=True, alternative="greater")

meanControlComparisonSpeedAssessmentToHH1= mean(ControlComparisonSpeedAssessmentToHH1)
pstdevControlComparisonSpeedAssessmentToHH1= pstdev(ControlComparisonSpeedAssessmentToHH1)

meanControlComparisonSpeedTestingToHH1= mean(ControlComparisonSpeedTestingToHH1)
pstdevControlComparisonSpeedTestingToHH1= pstdev(ControlComparisonSpeedTestingToHH1)

differenceBtwAssessSpeedsControlToHH1= meanControlComparisonSpeedAssessmentToHH1- meanControlComparisonSpeedTestingToHH1
fractionalDifferenceBtwAssessSpeedsControlToHH1= differenceBtwAssessSpeedsControlToHH1/meanControlComparisonSpeedAssessmentToHH1

ToHControlComparisonSpeedH1.insert(0, "Name", "Speed's Stats BTW Assessment and Testing ToH in H1 of Control Comparison")
ToHControlComparisonSpeedH1.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonSpeedAssessmentToHH1)
ToHControlComparisonSpeedH1.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonSpeedAssessmentToHH1)
ToHControlComparisonSpeedH1.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonSpeedTestingToHH1)
ToHControlComparisonSpeedH1.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonSpeedTestingToHH1)
ToHControlComparisonSpeedH1.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsControlToHH1)
ToHControlComparisonSpeedH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsControlToHH1)



# Stats Sorted and Grouped by Primary Task by H2
# All 60 datapoints per metric per participant per hypothesis within the Experimental Intervention

ExperimentalInterventionResumptionLagsAssessmentPathTracingH2=\
ExpH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment+\
ExpH2MathStroopMathDrawCollectedSumResumptionLagsAssessment

ExperimentalInterventionResumptionLagsAssessmentToHH2=\
ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment+\
ExpH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment

ExperimentalInterventionResumptionLagsTestingPathTracingH2=\
ExpH2StroopMathStroopDrawCollectedSumResumptionLagsTesting+\
ExpH2MathStroopMathDrawCollectedSumResumptionLagsTesting

ExperimentalInterventionResumptionLagsTestingToHH2=\
ExpH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting+\
ExpH2MathStroopMathHanoiCollectedSumResumptionLagsTesting

ExperimentalInterventionInterruptionLagsAssessmentPathTracingH2=\
ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment+\
ExpH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment

ExperimentalInterventionInterruptionLagsAssessmentToHH2=\
ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment+\
ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment

ExperimentalInterventionInterruptionLagsTestingPathTracingH2=\
ExpH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting+\
ExpH2MathStroopMathDrawCollectedSumInterruptionLagsTesting

ExperimentalInterventionInterruptionLagsTestingToHH2=\
ExpH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting+\
ExpH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting

ExperimentalInterventionAccuraciesAssessmentPathTracingH2=\
ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment+\
ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment

ExperimentalInterventionAccuraciesAssessmentToHH2=\
ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment+\
ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment

ExperimentalInterventionAccuraciesTestingPathTracingH2=\
ExpH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting+\
ExpH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting

ExperimentalInterventionAccuraciesTestingToHH2=\
ExpH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting+\
ExpH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting

ExperimentalInterventionSpeedAssessmentPathTracingH2= \
ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment+\
ExpH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment

ExperimentalInterventionSpeedAssessmentToHH2= \
ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment+\
ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment

ExperimentalInterventionSpeedTestingPathTracingH2= \
ExpH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting + \
ExpH2MathStroopMathDrawCollectedSumsCompletionTimesTesting

ExperimentalInterventionSpeedTestingToHH2= \
ExpH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting + \
ExpH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting


# All 60 datapoints per metric per participant per hypothesis within the Control Comparison
ControlComparisonResumptionLagsAssessmentPathTracingH2=\
ControlH2StroopMathStroopDrawCollectedSumResumptionLagsAssessment+\
ControlH2MathStroopMathDrawCollectedSumResumptionLagsAssessment

ControlComparisonResumptionLagsAssessmentToHH2=\
ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsAssessment+\
ControlH2MathStroopMathHanoiCollectedSumResumptionLagsAssessment

ControlComparisonResumptionLagsTestingPathTracingH2=\
ControlH2StroopMathStroopDrawCollectedSumResumptionLagsTesting+\
ControlH2MathStroopMathDrawCollectedSumResumptionLagsTesting

ControlComparisonResumptionLagsTestingToHH2=\
ControlH2StroopMathStroopHanoiCollectedSumResumptionLagsTesting+\
ControlH2MathStroopMathHanoiCollectedSumResumptionLagsTesting

ControlComparisonInterruptionLagsAssessmentPathTracingH2=\
ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsAssessment+\
ControlH2MathStroopMathDrawCollectedSumInterruptionLagsAssessment

ControlComparisonInterruptionLagsAssessmentToHH2=\
ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsAssessment+\
ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsAssessment

ControlComparisonInterruptionLagsTestingPathTracingH2=\
ControlH2StroopMathStroopDrawCollectedSumInterruptionLagsTesting+\
ControlH2MathStroopMathDrawCollectedSumInterruptionLagsTesting

ControlComparisonInterruptionLagsTestingToHH2=\
ControlH2StroopMathStroopHanoiCollectedSumInterruptionLagsTesting+\
ControlH2MathStroopMathHanoiCollectedSumInterruptionLagsTesting

ControlComparisonAccuraciesAssessmentPathTracingH2=\
ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesAssessment+\
ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesAssessment

ControlComparisonAccuraciesAssessmentToHH2=\
ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesAssessment+\
ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesAssessment

ControlComparisonAccuraciesTestingPathTracingH2=\
ControlH2StroopMathStroopDrawCollectedSumsMovesAndSequencesTesting+\
ControlH2MathStroopMathDrawCollectedSumsMovesAndSequencesTesting

ControlComparisonAccuraciesTestingToHH2=\
ControlH2StroopMathStroopHanoiCollectedSumsMovesAndSequencesTesting+\
ControlH2MathStroopMathHanoiCollectedSumsMovesAndSequencesTesting

ControlComparisonSpeedAssessmentPathTracingH2= \
ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesAssessment+\
ControlH2MathStroopMathDrawCollectedSumsCompletionTimesAssessment

ControlComparisonSpeedAssessmentToHH2= \
ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesAssessment+\
ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesAssessment

ControlComparisonSpeedTestingPathTracingH2= \
ControlH2StroopMathStroopDrawCollectedSumsCompletionTimesTesting + \
ControlH2MathStroopMathDrawCollectedSumsCompletionTimesTesting

ControlComparisonSpeedTestingToHH2= \
ControlH2StroopMathStroopHanoiCollectedSumsCompletionTimesTesting + \
ControlH2MathStroopMathHanoiCollectedSumsCompletionTimesTesting


space = [""]
AssessTestingPathTracingH2ExperimentalHeader = pd.DataFrame(data={"": space})
AssessTestingPathTracingH2ExperimentalHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for Path-Tracing in H2 of Experimental Intervention")

space = [""]
AssessTestingInToHH2ExperimentalHeader = pd.DataFrame(data={"": space})
AssessTestingInToHH2ExperimentalHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for ToH in H2 of Experimental Intervention")

space = [""]
AssessTestingPathTracingH2ControlHeader = pd.DataFrame(data={"": space})
AssessTestingPathTracingH2ControlHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for Path-Tracing in H2 of Control Comparison")

space = [""]
AssessTestingToHH2ControlHeader = pd.DataFrame(data={"": space})
AssessTestingToHH2ControlHeader.insert(0, "Name", "Metrics Stats BTW Assessment and Testing for ToH in H2 of Control Comparison")


# Path Tracing of Experimental Intervention Resumption Stats
PathTracingExperimentalInterventionResumptionLagsH2 = statisticize.ttest(ExperimentalInterventionResumptionLagsAssessmentPathTracingH2,
                                                 ExperimentalInterventionResumptionLagsTestingPathTracingH2, paired=True, alternative="greater")

meanExperimentalInterventionResumptionLagsAssessmentPathTracingH2 = mean(ExperimentalInterventionResumptionLagsAssessmentPathTracingH2)
pstdevExperimentalInterventionResumptionLagsAssessmentPathTracingH2 = pstdev(ExperimentalInterventionResumptionLagsAssessmentPathTracingH2)

meanExperimentalInterventionResumptionLagsTestingPathTracingH2 = mean(ExperimentalInterventionResumptionLagsTestingPathTracingH2)
pstdevExperimentalInterventionResumptionLagsTestingPathTracingH2 = pstdev(ExperimentalInterventionResumptionLagsTestingPathTracingH2)

differenceBtwAssessResumePathTracingH2 = meanExperimentalInterventionResumptionLagsAssessmentPathTracingH2 - meanExperimentalInterventionResumptionLagsTestingPathTracingH2
fractionalDifferenceBtwAssessResumePathTracingH2 = differenceBtwAssessResumePathTracingH2/meanExperimentalInterventionResumptionLagsAssessmentPathTracingH2

PathTracingExperimentalInterventionResumptionLagsH2.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Path Tracing of Experimental Intervention H2")
PathTracingExperimentalInterventionResumptionLagsH2.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionResumptionLagsAssessmentPathTracingH2)
PathTracingExperimentalInterventionResumptionLagsH2.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionResumptionLagsAssessmentPathTracingH2)
PathTracingExperimentalInterventionResumptionLagsH2.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionResumptionLagsTestingPathTracingH2)
PathTracingExperimentalInterventionResumptionLagsH2.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionResumptionLagsTestingPathTracingH2)
PathTracingExperimentalInterventionResumptionLagsH2.insert(13, "Diff In Seconds", differenceBtwAssessResumePathTracingH2)
PathTracingExperimentalInterventionResumptionLagsH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumePathTracingH2)


# ToH of Experimental Intervention Resumption Stats
ToHExperimentalInterventionResumptionLagsH2 = statisticize.ttest(ExperimentalInterventionResumptionLagsAssessmentToHH2,
                                                 ExperimentalInterventionResumptionLagsTestingToHH2, paired=True, alternative="greater")

meanExperimentalInterventionResumptionLagsAssessmentToHH2= mean(ExperimentalInterventionResumptionLagsAssessmentToHH2)
pstdevExperimentalInterventionResumptionLagsAssessmentToHH2= pstdev(ExperimentalInterventionResumptionLagsAssessmentToHH2)

meanExperimentalInterventionResumptionLagsTestingToHH2= mean(ExperimentalInterventionResumptionLagsTestingToHH2)
pstdevExperimentalInterventionResumptionLagsTestingToHH2= pstdev(ExperimentalInterventionResumptionLagsTestingToHH2)

differenceBtwAssessResumeToHH2= meanExperimentalInterventionResumptionLagsAssessmentToHH2- meanExperimentalInterventionResumptionLagsTestingToHH2
fractionalDifferenceBtwAssessResumeToHH2= differenceBtwAssessResumeToHH2/meanExperimentalInterventionResumptionLagsAssessmentToHH2

ToHExperimentalInterventionResumptionLagsH2.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing ToH of Experimental Intervention")
ToHExperimentalInterventionResumptionLagsH2.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionResumptionLagsAssessmentToHH2)
ToHExperimentalInterventionResumptionLagsH2.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionResumptionLagsAssessmentToHH2)
ToHExperimentalInterventionResumptionLagsH2.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionResumptionLagsTestingToHH2)
ToHExperimentalInterventionResumptionLagsH2.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionResumptionLagsTestingToHH2)
ToHExperimentalInterventionResumptionLagsH2.insert(13, "Diff In Seconds", differenceBtwAssessResumeToHH2)
ToHExperimentalInterventionResumptionLagsH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeToHH2)


# Path Tracing of Experimental Intervention Interruption Stats
PathTracingExperimentalInterventionInterruptionLagsH2 = statisticize.ttest(ExperimentalInterventionInterruptionLagsAssessmentPathTracingH2,
                                                 ExperimentalInterventionInterruptionLagsTestingPathTracingH2, paired=True, alternative="greater")

meanExperimentalInterventionInterruptionLagsAssessmentPathTracingH2 = mean(ExperimentalInterventionInterruptionLagsAssessmentPathTracingH2)
pstdevExperimentalInterventionInterruptionLagsAssessmentPathTracingH2 = pstdev(ExperimentalInterventionInterruptionLagsAssessmentPathTracingH2)

meanExperimentalInterventionInterruptionLagsTestingPathTracingH2 = mean(ExperimentalInterventionInterruptionLagsTestingPathTracingH2)
pstdevExperimentalInterventionInterruptionLagsTestingPathTracingH2 = pstdev(ExperimentalInterventionInterruptionLagsTestingPathTracingH2)

differenceBtwAssessAttendPathTracingH2 = meanExperimentalInterventionInterruptionLagsAssessmentPathTracingH2 - meanExperimentalInterventionInterruptionLagsTestingPathTracingH2
fractionalDifferenceBtwAssessAttendPathTracingH2 = differenceBtwAssessAttendPathTracingH2/meanExperimentalInterventionInterruptionLagsAssessmentPathTracingH2

PathTracingExperimentalInterventionInterruptionLagsH2.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Path Tracing of Experimental Intervention")
PathTracingExperimentalInterventionInterruptionLagsH2.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionInterruptionLagsAssessmentPathTracingH2)
PathTracingExperimentalInterventionInterruptionLagsH2.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionInterruptionLagsAssessmentPathTracingH2)
PathTracingExperimentalInterventionInterruptionLagsH2.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionInterruptionLagsTestingPathTracingH2)
PathTracingExperimentalInterventionInterruptionLagsH2.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionInterruptionLagsTestingPathTracingH2)
PathTracingExperimentalInterventionInterruptionLagsH2.insert(13, "Diff In Seconds", differenceBtwAssessAttendPathTracingH2)
PathTracingExperimentalInterventionInterruptionLagsH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendPathTracingH2)


# ToH of Experimental Intervention Interruption Stats
ToHExperimentalInterventionInterruptionLagsH2 = statisticize.ttest(ExperimentalInterventionInterruptionLagsAssessmentToHH2,
                                                 ExperimentalInterventionInterruptionLagsTestingToHH2, paired=True, alternative="greater")

meanExperimentalInterventionInterruptionLagsAssessmentToHH2= mean(ExperimentalInterventionInterruptionLagsAssessmentToHH2)
pstdevExperimentalInterventionInterruptionLagsAssessmentToHH2= pstdev(ExperimentalInterventionInterruptionLagsAssessmentToHH2)

meanExperimentalInterventionInterruptionLagsTestingToHH2= mean(ExperimentalInterventionInterruptionLagsTestingToHH2)
pstdevExperimentalInterventionInterruptionLagsTestingToHH2= pstdev(ExperimentalInterventionInterruptionLagsTestingToHH2)

differenceBtwAssessAttendToHH2= meanExperimentalInterventionInterruptionLagsAssessmentToHH2- meanExperimentalInterventionInterruptionLagsTestingToHH2
fractionalDifferenceBtwAssessAttendToHH2= differenceBtwAssessAttendToHH2/meanExperimentalInterventionInterruptionLagsAssessmentToHH2

ToHExperimentalInterventionInterruptionLagsH2.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing ToH of Experimental Intervention")
ToHExperimentalInterventionInterruptionLagsH2.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionInterruptionLagsAssessmentToHH2)
ToHExperimentalInterventionInterruptionLagsH2.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionInterruptionLagsAssessmentToHH2)
ToHExperimentalInterventionInterruptionLagsH2.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionInterruptionLagsTestingToHH2)
ToHExperimentalInterventionInterruptionLagsH2.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionInterruptionLagsTestingToHH2)
ToHExperimentalInterventionInterruptionLagsH2.insert(13, "Diff In Seconds", differenceBtwAssessAttendToHH2)
ToHExperimentalInterventionInterruptionLagsH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendToHH2)


# Path Tracing of Experimental Intervention Accuracy Stats
PathTracingExperimentalInterventionAccuraciesH2 = statisticize.ttest(ExperimentalInterventionAccuraciesAssessmentPathTracingH2,
                                                 ExperimentalInterventionAccuraciesTestingPathTracingH2, paired=True, alternative="greater")

meanExperimentalInterventionAccuraciesAssessmentPathTracingH2 = mean(ExperimentalInterventionAccuraciesAssessmentPathTracingH2)
pstdevExperimentalInterventionAccuraciesAssessmentPathTracingH2 = pstdev(ExperimentalInterventionAccuraciesAssessmentPathTracingH2)

meanExperimentalInterventionAccuraciesTestingPathTracingH2 = mean(ExperimentalInterventionAccuraciesTestingPathTracingH2)
pstdevExperimentalInterventionAccuraciesTestingPathTracingH2 = pstdev(ExperimentalInterventionAccuraciesTestingPathTracingH2)

differenceBtwAssessAccuraciesPathTracingH2 = meanExperimentalInterventionAccuraciesAssessmentPathTracingH2 - meanExperimentalInterventionAccuraciesTestingPathTracingH2
fractionalDifferenceBtwAssessAccuraciesPathTracingH2 = differenceBtwAssessAccuraciesPathTracingH2

PathTracingExperimentalInterventionAccuraciesH2.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Path Tracing of Experimental Intervention")
PathTracingExperimentalInterventionAccuraciesH2.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionAccuraciesAssessmentPathTracingH2)
PathTracingExperimentalInterventionAccuraciesH2.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionAccuraciesAssessmentPathTracingH2)
PathTracingExperimentalInterventionAccuraciesH2.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionAccuraciesTestingPathTracingH2)
PathTracingExperimentalInterventionAccuraciesH2.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionAccuraciesTestingPathTracingH2)
PathTracingExperimentalInterventionAccuraciesH2.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesPathTracingH2)
PathTracingExperimentalInterventionAccuraciesH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesPathTracingH2)


# ToH of Experimental Intervention Accuracy Stats
ToHExperimentalInterventionAccuraciesH2 = statisticize.ttest(ExperimentalInterventionAccuraciesAssessmentToHH2,
                                                 ExperimentalInterventionAccuraciesTestingToHH2, paired=True, alternative="greater")

meanExperimentalInterventionAccuraciesAssessmentToHH2= mean(ExperimentalInterventionAccuraciesAssessmentToHH2)
pstdevExperimentalInterventionAccuraciesAssessmentToHH2= pstdev(ExperimentalInterventionAccuraciesAssessmentToHH2)

meanExperimentalInterventionAccuraciesTestingToHH2= mean(ExperimentalInterventionAccuraciesTestingToHH2)
pstdevExperimentalInterventionAccuraciesTestingToHH2= pstdev(ExperimentalInterventionAccuraciesTestingToHH2)

differenceBtwAssessAccuraciesToHH2= meanExperimentalInterventionAccuraciesAssessmentToHH2- meanExperimentalInterventionAccuraciesTestingToHH2
fractionalDifferenceBtwAssessAccuraciesToHH2= differenceBtwAssessAccuraciesToHH2

ToHExperimentalInterventionAccuraciesH2.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing ToH of Experimental Intervention")
ToHExperimentalInterventionAccuraciesH2.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionAccuraciesAssessmentToHH2)
ToHExperimentalInterventionAccuraciesH2.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionAccuraciesAssessmentToHH2)
ToHExperimentalInterventionAccuraciesH2.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionAccuraciesTestingToHH2)
ToHExperimentalInterventionAccuraciesH2.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionAccuraciesTestingToHH2)
ToHExperimentalInterventionAccuraciesH2.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesToHH2)
ToHExperimentalInterventionAccuraciesH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesToHH2)


# Path Tracing of Experimental Intervention Speed Stats
PathTracingExperimentalInterventionSpeedH2 = statisticize.ttest(ExperimentalInterventionSpeedAssessmentPathTracingH2,
                                                 ExperimentalInterventionSpeedTestingPathTracingH2, paired=True, alternative="greater")

meanExperimentalInterventionSpeedAssessmentPathTracingH2 = mean(ExperimentalInterventionSpeedAssessmentPathTracingH2)
pstdevExperimentalInterventionSpeedAssessmentPathTracingH2 = pstdev(ExperimentalInterventionSpeedAssessmentPathTracingH2)

meanExperimentalInterventionSpeedTestingPathTracingH2 = mean(ExperimentalInterventionSpeedTestingPathTracingH2)
pstdevExperimentalInterventionSpeedTestingPathTracingH2 = pstdev(ExperimentalInterventionSpeedTestingPathTracingH2)

differenceBtwAssessSpeedsPathTracingH2 = meanExperimentalInterventionSpeedAssessmentPathTracingH2 - meanExperimentalInterventionSpeedTestingPathTracingH2
fractionalDifferenceBtwAssessSpeedsPathTracingH2 = differenceBtwAssessSpeedsPathTracingH2/meanExperimentalInterventionSpeedAssessmentPathTracingH2

PathTracingExperimentalInterventionSpeedH2.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Path Tracing of Experimental Intervention")
PathTracingExperimentalInterventionSpeedH2.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionSpeedAssessmentPathTracingH2)
PathTracingExperimentalInterventionSpeedH2.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionSpeedAssessmentPathTracingH2)
PathTracingExperimentalInterventionSpeedH2.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionSpeedTestingPathTracingH2)
PathTracingExperimentalInterventionSpeedH2.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionSpeedTestingPathTracingH2)
PathTracingExperimentalInterventionSpeedH2.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsPathTracingH2)
PathTracingExperimentalInterventionSpeedH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsPathTracingH2)

# ToH of Experimental Intervention Speed Stats
ToHExperimentalInterventionSpeedH2 = statisticize.ttest(ExperimentalInterventionSpeedAssessmentToHH2,
                                                 ExperimentalInterventionSpeedTestingToHH2, paired=True, alternative="greater")

meanExperimentalInterventionSpeedAssessmentToHH2= mean(ExperimentalInterventionSpeedAssessmentToHH2)
pstdevExperimentalInterventionSpeedAssessmentToHH2= pstdev(ExperimentalInterventionSpeedAssessmentToHH2)

meanExperimentalInterventionSpeedTestingToHH2= mean(ExperimentalInterventionSpeedTestingToHH2)
pstdevExperimentalInterventionSpeedTestingToHH2= pstdev(ExperimentalInterventionSpeedTestingToHH2)

differenceBtwAssessSpeedsToHH2= meanExperimentalInterventionSpeedAssessmentToHH2- meanExperimentalInterventionSpeedTestingToHH2
fractionalDifferenceBtwAssessSpeedsToHH2= differenceBtwAssessSpeedsToHH2/meanExperimentalInterventionSpeedAssessmentToHH2

ToHExperimentalInterventionSpeedH2.insert(0, "Name", "Speed's Stats BTW Assessment and Testing ToH of Experimental Intervention")
ToHExperimentalInterventionSpeedH2.insert(9, "Mean Sample 1 (Seconds)", meanExperimentalInterventionSpeedAssessmentToHH2)
ToHExperimentalInterventionSpeedH2.insert(10, "SD Sample 1 (Seconds)", pstdevExperimentalInterventionSpeedAssessmentToHH2)
ToHExperimentalInterventionSpeedH2.insert(11, "Mean Sample 2 (Seconds)", meanExperimentalInterventionSpeedTestingToHH2)
ToHExperimentalInterventionSpeedH2.insert(12, "SD Sample 2 (Seconds)", pstdevExperimentalInterventionSpeedTestingToHH2)
ToHExperimentalInterventionSpeedH2.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsToHH2)
ToHExperimentalInterventionSpeedH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsToHH2)



# Control Comparison Stats
# Path Tracing of Control Comparison Resumption Stats
PathTracingControlComparisonResumptionLagsH2 = statisticize.ttest(ControlComparisonResumptionLagsAssessmentPathTracingH2,
                                                 ControlComparisonResumptionLagsTestingPathTracingH2, paired=True, alternative="greater")

meanControlComparisonResumptionLagsAssessmentPathTracingH2 = mean(ControlComparisonResumptionLagsAssessmentPathTracingH2)
pstdevControlComparisonResumptionLagsAssessmentPathTracingH2 = pstdev(ControlComparisonResumptionLagsAssessmentPathTracingH2)

meanControlComparisonResumptionLagsTestingPathTracingH2 = mean(ControlComparisonResumptionLagsTestingPathTracingH2)
pstdevControlComparisonResumptionLagsTestingPathTracingH2 = pstdev(ControlComparisonResumptionLagsTestingPathTracingH2)

differenceBtwAssessResumeControlPathTracingH2 = meanControlComparisonResumptionLagsAssessmentPathTracingH2 - meanControlComparisonResumptionLagsTestingPathTracingH2
fractionalDifferenceBtwAssessResumeControlPathTracingH2 = differenceBtwAssessResumeControlPathTracingH2/meanControlComparisonResumptionLagsAssessmentPathTracingH2

PathTracingControlComparisonResumptionLagsH2.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing Path Tracing Control Comparison")
PathTracingControlComparisonResumptionLagsH2.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonResumptionLagsAssessmentPathTracingH2)
PathTracingControlComparisonResumptionLagsH2.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonResumptionLagsAssessmentPathTracingH2)
PathTracingControlComparisonResumptionLagsH2.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonResumptionLagsTestingPathTracingH2)
PathTracingControlComparisonResumptionLagsH2.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonResumptionLagsTestingPathTracingH2)
PathTracingControlComparisonResumptionLagsH2.insert(13, "Diff In Seconds", differenceBtwAssessResumeControlPathTracingH2)
PathTracingControlComparisonResumptionLagsH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeControlPathTracingH2)


# To Hof Control Comparison Resumption Stats
ToHControlComparisonResumptionLagsH2 = statisticize.ttest(ControlComparisonResumptionLagsAssessmentToHH2,
                                                 ControlComparisonResumptionLagsTestingToHH2, paired=True, alternative="greater")

meanControlComparisonResumptionLagsAssessmentToHH2= mean(ControlComparisonResumptionLagsAssessmentToHH2)
pstdevControlComparisonResumptionLagsAssessmentToHH2= pstdev(ControlComparisonResumptionLagsAssessmentToHH2)

meanControlComparisonResumptionLagsTestingToHH2= mean(ControlComparisonResumptionLagsTestingToHH2)
pstdevControlComparisonResumptionLagsTestingToHH2= pstdev(ControlComparisonResumptionLagsTestingToHH2)

differenceBtwAssessResumeControlToHH2= meanControlComparisonResumptionLagsAssessmentToHH2- meanControlComparisonResumptionLagsTestingToHH2
fractionalDifferenceBtwAssessResumeControlToHH2= differenceBtwAssessResumeControlToHH2/meanControlComparisonResumptionLagsAssessmentToHH2

ToHControlComparisonResumptionLagsH2.insert(0, "Name", "Resumption Lag's Stats BTW Assessment and Testing ToH Control Comparison")
ToHControlComparisonResumptionLagsH2.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonResumptionLagsAssessmentToHH2)
ToHControlComparisonResumptionLagsH2.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonResumptionLagsAssessmentToHH2)
ToHControlComparisonResumptionLagsH2.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonResumptionLagsTestingToHH2)
ToHControlComparisonResumptionLagsH2.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonResumptionLagsTestingToHH2)
ToHControlComparisonResumptionLagsH2.insert(13, "Diff In Seconds", differenceBtwAssessResumeControlToHH2)
ToHControlComparisonResumptionLagsH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessResumeControlToHH2)


# Path Tracing of Control Comparison Interruption Stats
PathTracingControlComparisonInterruptionLagsH2 = statisticize.ttest(ControlComparisonInterruptionLagsAssessmentPathTracingH2,
                                                 ControlComparisonInterruptionLagsTestingPathTracingH2, paired=True, alternative="greater")

meanControlComparisonInterruptionLagsAssessmentPathTracingH2 = mean(ControlComparisonInterruptionLagsAssessmentPathTracingH2)
pstdevControlComparisonInterruptionLagsAssessmentPathTracingH2 = pstdev(ControlComparisonInterruptionLagsAssessmentPathTracingH2)

meanControlComparisonInterruptionLagsTestingPathTracingH2 = mean(ControlComparisonInterruptionLagsTestingPathTracingH2)
pstdevControlComparisonInterruptionLagsTestingPathTracingH2 = pstdev(ControlComparisonInterruptionLagsTestingPathTracingH2)

differenceBtwAssessAttendControlPathTracingH2 = meanControlComparisonInterruptionLagsAssessmentPathTracingH2 - meanControlComparisonInterruptionLagsTestingPathTracingH2
fractionalDifferenceBtwAssessAttendControlPathTracingH2 = differenceBtwAssessAttendControlPathTracingH2/meanControlComparisonInterruptionLagsAssessmentPathTracingH2

PathTracingControlComparisonInterruptionLagsH2.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing Path Tracing H2 Control Comparison")
PathTracingControlComparisonInterruptionLagsH2.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonInterruptionLagsAssessmentPathTracingH2)
PathTracingControlComparisonInterruptionLagsH2.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonInterruptionLagsAssessmentPathTracingH2)
PathTracingControlComparisonInterruptionLagsH2.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonInterruptionLagsTestingPathTracingH2)
PathTracingControlComparisonInterruptionLagsH2.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonInterruptionLagsTestingPathTracingH2)
PathTracingControlComparisonInterruptionLagsH2.insert(13, "Diff In Seconds", differenceBtwAssessAttendControlPathTracingH2)
PathTracingControlComparisonInterruptionLagsH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendControlPathTracingH2)


# ToH of Control Comparison Interruption Stats
ToHControlComparisonInterruptionLagsH2 = statisticize.ttest(ControlComparisonInterruptionLagsAssessmentToHH2,
                                                 ControlComparisonInterruptionLagsTestingToHH2, paired=True, alternative="greater")

meanControlComparisonInterruptionLagsAssessmentToHH2= mean(ControlComparisonInterruptionLagsAssessmentToHH2)
pstdevControlComparisonInterruptionLagsAssessmentToHH2= pstdev(ControlComparisonInterruptionLagsAssessmentToHH2)

meanControlComparisonInterruptionLagsTestingToHH2= mean(ControlComparisonInterruptionLagsTestingToHH2)
pstdevControlComparisonInterruptionLagsTestingToHH2= pstdev(ControlComparisonInterruptionLagsTestingToHH2)

differenceBtwAssessAttendControlToHH2= meanControlComparisonInterruptionLagsAssessmentToHH2- meanControlComparisonInterruptionLagsTestingToHH2
fractionalDifferenceBtwAssessAttendControlToHH2= differenceBtwAssessAttendControlToHH2/meanControlComparisonInterruptionLagsAssessmentToHH2

ToHControlComparisonInterruptionLagsH2.insert(0, "Name", "Interruption Lag's Stats BTW Assessment and Testing ToH H2 Control Comparison")
ToHControlComparisonInterruptionLagsH2.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonInterruptionLagsAssessmentToHH2)
ToHControlComparisonInterruptionLagsH2.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonInterruptionLagsAssessmentToHH2)
ToHControlComparisonInterruptionLagsH2.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonInterruptionLagsTestingToHH2)
ToHControlComparisonInterruptionLagsH2.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonInterruptionLagsTestingToHH2)
ToHControlComparisonInterruptionLagsH2.insert(13, "Diff In Seconds", differenceBtwAssessAttendControlToHH2)
ToHControlComparisonInterruptionLagsH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAttendControlToHH2)


# Path Tracing of Control Comparison Accuracy Stats
PathTracingControlComparisonAccuraciesH2 = statisticize.ttest(ControlComparisonAccuraciesAssessmentPathTracingH2,
                                                 ControlComparisonAccuraciesTestingPathTracingH2, paired=True, alternative="greater")

meanControlComparisonAccuraciesAssessmentPathTracingH2 = mean(ControlComparisonAccuraciesAssessmentPathTracingH2)
pstdevControlComparisonAccuraciesAssessmentPathTracingH2 = pstdev(ControlComparisonAccuraciesAssessmentPathTracingH2)

meanControlComparisonAccuraciesTestingPathTracingH2 = mean(ControlComparisonAccuraciesTestingPathTracingH2)
pstdevControlComparisonAccuraciesTestingPathTracingH2 = pstdev(ControlComparisonAccuraciesTestingPathTracingH2)

differenceBtwAssessAccuraciesControlPathTracingH2 = meanControlComparisonAccuraciesAssessmentPathTracingH2 - meanControlComparisonAccuraciesTestingPathTracingH2
fractionalDifferenceBtwAssessAccuraciesControlPathTracingH2 = differenceBtwAssessAccuraciesControlPathTracingH2

PathTracingControlComparisonAccuraciesH2.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing Path Tracing H2 Control Comparison")
PathTracingControlComparisonAccuraciesH2.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonAccuraciesAssessmentPathTracingH2)
PathTracingControlComparisonAccuraciesH2.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonAccuraciesAssessmentPathTracingH2)
PathTracingControlComparisonAccuraciesH2.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonAccuraciesTestingPathTracingH2)
PathTracingControlComparisonAccuraciesH2.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonAccuraciesTestingPathTracingH2)
PathTracingControlComparisonAccuraciesH2.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesControlPathTracingH2)
PathTracingControlComparisonAccuraciesH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesControlPathTracingH2)


# ToH of Control Comparison Accuracy Stats
ToHControlComparisonAccuraciesH2 = statisticize.ttest(ControlComparisonAccuraciesAssessmentToHH2,
                                                 ControlComparisonAccuraciesTestingToHH2, paired=True, alternative="greater")

meanControlComparisonAccuraciesAssessmentToHH2= mean(ControlComparisonAccuraciesAssessmentToHH2)
pstdevControlComparisonAccuraciesAssessmentToHH2= pstdev(ControlComparisonAccuraciesAssessmentToHH2)

meanControlComparisonAccuraciesTestingToHH2= mean(ControlComparisonAccuraciesTestingToHH2)
pstdevControlComparisonAccuraciesTestingToHH2= pstdev(ControlComparisonAccuraciesTestingToHH2)

differenceBtwAssessAccuraciesControlToHH2= meanControlComparisonAccuraciesAssessmentToHH2- meanControlComparisonAccuraciesTestingToHH2
fractionalDifferenceBtwAssessAccuraciesControlToHH2= differenceBtwAssessAccuraciesControlToHH2

ToHControlComparisonAccuraciesH2.insert(0, "Name", "Accuracies' Stats BTW Assessment and Testing ToH H2 Control Comparison")
ToHControlComparisonAccuraciesH2.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonAccuraciesAssessmentToHH2)
ToHControlComparisonAccuraciesH2.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonAccuraciesAssessmentToHH2)
ToHControlComparisonAccuraciesH2.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonAccuraciesTestingToHH2)
ToHControlComparisonAccuraciesH2.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonAccuraciesTestingToHH2)
ToHControlComparisonAccuraciesH2.insert(13, "Diff In Seconds", differenceBtwAssessAccuraciesControlToHH2)
ToHControlComparisonAccuraciesH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessAccuraciesControlToHH2)


# Path Tracing of Control Comparison Speed Stats
PathTracingControlComparisonSpeedH2 = statisticize.ttest(ControlComparisonSpeedAssessmentPathTracingH2,
                                                 ControlComparisonSpeedTestingPathTracingH2, paired=True, alternative="greater")

meanControlComparisonSpeedAssessmentPathTracingH2 = mean(ControlComparisonSpeedAssessmentPathTracingH2)
pstdevControlComparisonSpeedAssessmentPathTracingH2 = pstdev(ControlComparisonSpeedAssessmentPathTracingH2)

meanControlComparisonSpeedTestingPathTracingH2 = mean(ControlComparisonSpeedTestingPathTracingH2)
pstdevControlComparisonSpeedTestingPathTracingH2 = pstdev(ControlComparisonSpeedTestingPathTracingH2)

differenceBtwAssessSpeedsControlPathTracingH2 = meanControlComparisonSpeedAssessmentPathTracingH2 - meanControlComparisonSpeedTestingPathTracingH2
fractionalDifferenceBtwAssessSpeedsControlPathTracingH2 = differenceBtwAssessSpeedsControlPathTracingH2/meanControlComparisonSpeedAssessmentPathTracingH2

PathTracingControlComparisonSpeedH2.insert(0, "Name", "Speed's Stats BTW Assessment and Testing Path Tracing Control Comparison")
PathTracingControlComparisonSpeedH2.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonSpeedAssessmentPathTracingH2)
PathTracingControlComparisonSpeedH2.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonSpeedAssessmentPathTracingH2)
PathTracingControlComparisonSpeedH2.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonSpeedTestingPathTracingH2)
PathTracingControlComparisonSpeedH2.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonSpeedTestingPathTracingH2)
PathTracingControlComparisonSpeedH2.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsControlPathTracingH2)
PathTracingControlComparisonSpeedH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsControlPathTracingH2)


# ToH of Control Comparison Speed Stats
ToHControlComparisonSpeedH2 = statisticize.ttest(ControlComparisonSpeedAssessmentToHH2,
                                                 ControlComparisonSpeedTestingToHH2, paired=True, alternative="greater")

meanControlComparisonSpeedAssessmentToHH2= mean(ControlComparisonSpeedAssessmentToHH2)
pstdevControlComparisonSpeedAssessmentToHH2= pstdev(ControlComparisonSpeedAssessmentToHH2)

meanControlComparisonSpeedTestingToHH2= mean(ControlComparisonSpeedTestingToHH2)
pstdevControlComparisonSpeedTestingToHH2= pstdev(ControlComparisonSpeedTestingToHH2)

differenceBtwAssessSpeedsControlToHH2= meanControlComparisonSpeedAssessmentToHH2- meanControlComparisonSpeedTestingToHH2
fractionalDifferenceBtwAssessSpeedsControlToHH2= differenceBtwAssessSpeedsControlToHH2/meanControlComparisonSpeedAssessmentToHH2

ToHControlComparisonSpeedH2.insert(0, "Name", "Speed's Stats BTW Assessment and Testing ToH Control Comparison")
ToHControlComparisonSpeedH2.insert(9, "Mean Sample 1 (Seconds)", meanControlComparisonSpeedAssessmentToHH2)
ToHControlComparisonSpeedH2.insert(10, "SD Sample 1 (Seconds)", pstdevControlComparisonSpeedAssessmentToHH2)
ToHControlComparisonSpeedH2.insert(11, "Mean Sample 2 (Seconds)", meanControlComparisonSpeedTestingToHH2)
ToHControlComparisonSpeedH2.insert(12, "SD Sample 2 (Seconds)", pstdevControlComparisonSpeedTestingToHH2)
ToHControlComparisonSpeedH2.insert(13, "Diff In Seconds", differenceBtwAssessSpeedsControlToHH2)
ToHControlComparisonSpeedH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessSpeedsControlToHH2)



# Error Rate/Accuracy/Correctness/Optimality
# -----------------------------------------------------------------
errorRateAssessmentH1 = \
    ExperimentalInterventionAccuraciesAssessmentPathTracingH1 + \
    ControlComparisonAccuraciesAssessmentPathTracingH1

errorRateTestingH1 = \
    ExperimentalInterventionAccuraciesTestingPathTracingH1 + \
    ControlComparisonAccuraciesTestingPathTracingH1

optimalityAssessmentH1 = \
    ExperimentalInterventionAccuraciesAssessmentToHH1 + \
    ControlComparisonAccuraciesAssessmentToHH1

optimalityTestingH1 = \
    ExperimentalInterventionAccuraciesTestingToHH1 + \
    ControlComparisonAccuraciesTestingToHH1



errorRateAssessmentH2 = \
    ExperimentalInterventionAccuraciesAssessmentPathTracingH2 + \
    ControlComparisonAccuraciesAssessmentPathTracingH2

errorRateTestingH2 = \
    ExperimentalInterventionAccuraciesTestingPathTracingH2 + \
    ControlComparisonAccuraciesTestingPathTracingH2

optimalityAssessmentH2 = \
    ExperimentalInterventionAccuraciesAssessmentToHH2 + \
    ControlComparisonAccuraciesAssessmentToHH2

optimalityTestingH2 = \
    ExperimentalInterventionAccuraciesTestingToHH2 + \
    ControlComparisonAccuraciesTestingToHH2

# Accuracy Both Hypotheses EXP plus CON
errorRateAssessmentH1H2 = errorRateAssessmentH1+errorRateAssessmentH2
errorRateTestingH1H2 = errorRateTestingH1+errorRateTestingH2

# Optimality Both Hypotheses EXP plus CON
optimalityAssessmentH1H2 = optimalityAssessmentH1+optimalityAssessmentH2
optimalityTestingH1H2 = optimalityTestingH1+optimalityTestingH2


# --------------------------------------------Accuracy of Path Tracing
accuracyBothHypothesesEXPplusCONStats = statisticize.ttest(errorRateAssessmentH1H2,
                                                 errorRateTestingH1H2, paired=True, alternative="greater")

meanAccBothHypothesesEXPplusCONAssess = mean(errorRateAssessmentH1H2)
pstdevAccBothHypothesesEXPplusCONAssess = pstdev(errorRateAssessmentH1H2)

meanAccBothHypothesesEXPplusCONTest = mean(errorRateTestingH1H2)
pstdevAccBothHypothesesEXPplusCONTest = pstdev(errorRateTestingH1H2)

differenceAccBothHypothesesEXPplusCONTest = meanAccBothHypothesesEXPplusCONAssess - meanAccBothHypothesesEXPplusCONTest
fractionalDifferenceAccBothHypothesesEXPplusCONTest = differenceAccBothHypothesesEXPplusCONTest

accuracyBothHypothesesEXPplusCONStats.insert(0, "Name", "Pre- vs Post Correctness across both Conditions (Path Tracing)")
accuracyBothHypothesesEXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanAccBothHypothesesEXPplusCONAssess)
accuracyBothHypothesesEXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevAccBothHypothesesEXPplusCONAssess)
accuracyBothHypothesesEXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanAccBothHypothesesEXPplusCONTest)
accuracyBothHypothesesEXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevAccBothHypothesesEXPplusCONTest)
accuracyBothHypothesesEXPplusCONStats.insert(13, "Diff In Seconds", differenceAccBothHypothesesEXPplusCONTest)
accuracyBothHypothesesEXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceAccBothHypothesesEXPplusCONTest)

# --------------------------------------------Optimality of Tower of Hanoi
optimalityBothHypothesesEXPplusCONStats = statisticize.ttest(optimalityAssessmentH1H2,
                                                 optimalityTestingH1H2, paired=True, alternative="greater")

meanOptBothHypothesesEXPplusCONAssess = mean(optimalityAssessmentH1H2)
pstdevOptBothHypothesesEXPplusCONAssess = pstdev(optimalityAssessmentH1H2)

meanOptBothHypothesesEXPplusCONTest = mean(optimalityTestingH1H2)
pstdevOptBothHypothesesEXPplusCONTest = pstdev(optimalityTestingH1H2)

differenceOptBothHypothesesEXPplusCONTest = meanOptBothHypothesesEXPplusCONAssess - meanOptBothHypothesesEXPplusCONTest
fractionalDifferenceOptBothHypothesesEXPplusCONTest = differenceOptBothHypothesesEXPplusCONTest

optimalityBothHypothesesEXPplusCONStats.insert(0, "Name", "Pre- vs Post Optimality across both Conditions (ToH)")
optimalityBothHypothesesEXPplusCONStats.insert(9, "Mean Sample 1 (Seconds)", meanOptBothHypothesesEXPplusCONAssess)
optimalityBothHypothesesEXPplusCONStats.insert(10, "SD Sample 1 (Seconds)", pstdevOptBothHypothesesEXPplusCONAssess)
optimalityBothHypothesesEXPplusCONStats.insert(11, "Mean Sample 2 (Seconds)", meanOptBothHypothesesEXPplusCONTest)
optimalityBothHypothesesEXPplusCONStats.insert(12, "SD Sample 2 (Seconds)", pstdevOptBothHypothesesEXPplusCONTest)
optimalityBothHypothesesEXPplusCONStats.insert(13, "Diff In Seconds", differenceOptBothHypothesesEXPplusCONTest)
optimalityBothHypothesesEXPplusCONStats.insert(14, "Fraction In Seconds", fractionalDifferenceOptBothHypothesesEXPplusCONTest)





# Individual Stats for each error-reduction performance metric H1

errorRatesBTWAssessTestH1 = statisticize.ttest(errorRateAssessmentH1, errorRateTestingH1, paired=True, alternative="greater")

meanErrorRateAssessmentH1= mean(errorRateAssessmentH1)
pstdevErrorRateAssessmentH1= pstdev(errorRateAssessmentH1)

meanErrorRateTestingH1= mean(errorRateTestingH1)
pstdevErrorRateTestingH1= pstdev(errorRateTestingH1)

differenceBtwAssessTestMeansPathH1= meanErrorRateAssessmentH1- meanErrorRateTestingH1
fractionalDifferenceBtwAssessTestMeansPathH1= differenceBtwAssessTestMeansPathH1

errorRatesBTWAssessTestH1.insert(0, "Name", "Path Tracing Accuracy's Stats BTW Assessment and Testing H1")
errorRatesBTWAssessTestH1.insert(9, "Mean Sample 1 (Seconds)", meanErrorRateAssessmentH1)
errorRatesBTWAssessTestH1.insert(10, "SD Sample 1 (Seconds)", pstdevErrorRateAssessmentH1)
errorRatesBTWAssessTestH1.insert(11, "Mean Sample 2 (Seconds)", meanErrorRateTestingH1)
errorRatesBTWAssessTestH1.insert(12, "SD Sample 2 (Seconds)", pstdevErrorRateTestingH1)
errorRatesBTWAssessTestH1.insert(13, "Diff In Seconds", differenceBtwAssessTestMeansPathH1)
errorRatesBTWAssessTestH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessTestMeansPathH1)


optimalityBTWAssessTestH1 = statisticize.ttest(optimalityAssessmentH1, optimalityTestingH1, paired=True, alternative="greater")

meanOptimalityAssessmentH1= mean(optimalityAssessmentH1)
pstdevOptimalityAssessmentH1= pstdev(optimalityAssessmentH1)

meanOptimalityTestingH1= mean(optimalityTestingH1)
pstdevOptimalityTestingH1= pstdev(optimalityTestingH1)

differenceBtwAssessTestMeansToHH1= meanOptimalityAssessmentH1- meanOptimalityTestingH1
fractionalDifferenceBtwAssessTestMeansToHH1= differenceBtwAssessTestMeansToHH1

optimalityBTWAssessTestH1.insert(0, "Name", "Hanoi's Optimality Stats BTW Assessment and Testing H1")
optimalityBTWAssessTestH1.insert(9, "Mean Sample 1 (Seconds)", meanOptimalityAssessmentH1)
optimalityBTWAssessTestH1.insert(10, "SD Sample 1 (Seconds)", pstdevOptimalityAssessmentH1)
optimalityBTWAssessTestH1.insert(11, "Mean Sample 2 (Seconds)", meanOptimalityTestingH1)
optimalityBTWAssessTestH1.insert(12, "SD Sample 2 (Seconds)", pstdevOptimalityTestingH1)
optimalityBTWAssessTestH1.insert(13, "Diff In Seconds", differenceBtwAssessTestMeansToHH1)
optimalityBTWAssessTestH1.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessTestMeansToHH1)



# Individual Stats for each error-reduction performance metric H2
errorRatesBTWAssessTestH2 = statisticize.ttest(errorRateAssessmentH2, errorRateTestingH2, paired=True, alternative="greater")

meanErrorRateAssessmentH2= mean(errorRateAssessmentH2)
pstdevErrorRateAssessmentH2= pstdev(errorRateAssessmentH2)

meanErrorRateTestingH2= mean(errorRateTestingH2)
pstdevErrorRateTestingH2= pstdev(errorRateTestingH2)

differenceBtwAssessTestMeansPathH2= meanErrorRateAssessmentH2- meanErrorRateTestingH2
fractionalDifferenceBtwAssessTestMeansPathH2= differenceBtwAssessTestMeansPathH2

errorRatesBTWAssessTestH2.insert(0, "Name", "Path Tracing Accuracy's Stats BTW Assessment and Testing H2")
errorRatesBTWAssessTestH2.insert(9, "Mean Sample 1 (Seconds)", meanErrorRateAssessmentH2)
errorRatesBTWAssessTestH2.insert(10, "SD Sample 1 (Seconds)", pstdevErrorRateAssessmentH2)
errorRatesBTWAssessTestH2.insert(11, "Mean Sample 2 (Seconds)", meanErrorRateTestingH2)
errorRatesBTWAssessTestH2.insert(12, "SD Sample 2 (Seconds)", pstdevErrorRateTestingH2)
errorRatesBTWAssessTestH2.insert(13, "Diff In Seconds", differenceBtwAssessTestMeansPathH2)
errorRatesBTWAssessTestH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessTestMeansPathH2)


optimalityBTWAssessTestH2 = statisticize.ttest(optimalityAssessmentH2, optimalityTestingH2, paired=True, alternative="greater")

meanOptimalityAssessmentH2= mean(optimalityAssessmentH2)
pstdevOptimalityAssessmentH2= pstdev(optimalityAssessmentH2)

meanOptimalityTestingH2= mean(optimalityTestingH2)
pstdevOptimalityTestingH2= pstdev(optimalityTestingH2)

differenceBtwAssessTestMeansToHH2= meanOptimalityAssessmentH2- meanOptimalityTestingH2
fractionalDifferenceBtwAssessTestMeansToHH2= differenceBtwAssessTestMeansToHH2

optimalityBTWAssessTestH2.insert(0, "Name", "Hanoi's Optimality Stats BTW Assessment and Testing H2")
optimalityBTWAssessTestH2.insert(9, "Mean Sample 1 (Seconds)", meanOptimalityAssessmentH2)
optimalityBTWAssessTestH2.insert(10, "SD Sample 1 (Seconds)", pstdevOptimalityAssessmentH2)
optimalityBTWAssessTestH2.insert(11, "Mean Sample 2 (Seconds)", meanOptimalityTestingH2)
optimalityBTWAssessTestH2.insert(12, "SD Sample 2 (Seconds)", pstdevOptimalityTestingH2)
optimalityBTWAssessTestH2.insert(13, "Diff In Seconds", differenceBtwAssessTestMeansToHH2)
optimalityBTWAssessTestH2.insert(14, "Fraction In Seconds", fractionalDifferenceBtwAssessTestMeansToHH2)


# Differences_BTW_AssessTest_1st and 4th quadrant EXP_Path_Errors
# Differences_BTW_AssessTest_CON_Path_Errors 2nd & 3rd quadrant
# Then stat each difference against eaxh other

SupportivePathErrorsH1H2 = \
ExpH1DrawHanoiDrawStroopDiffAccuracy + \
ExpH1DrawHanoiDrawMathDiffAccuracy + \
ExpH2StroopMathStroopDrawDiffAccuracy + \
ExpH2MathStroopMathDrawDiffAccuracy

AlternativePathErrorsH1H2 = \
ControlH1DrawHanoiDrawStroopDiffAccuracy + \
ControlH1DrawHanoiDrawMathDiffAccuracy + \
ControlH2StroopMathStroopDrawDiffAccuracy + \
ControlH2MathStroopMathDrawDiffAccuracy


# Differences_BTW_AssessTest_EXP_ToH_Errors
# Differences_BTW_AssessTest_CON_ToH_Errors
# Then stat each difference against eaxh other

SupportiveToHOptimalitiesH1H2 = \
ExpH1HanoiDrawHanoiStroopDiffAccuracy + \
ExpH1HanoiDrawHanoiMathDiffAccuracy + \
ExpH2StroopMathStroopHanoiDiffAccuracy + \
ExpH2MathStroopMathHanoiDiffAccuracy

AlternativeToHOptimalitiesH1H2 = \
ControlH1HanoiDrawHanoiStroopDiffAccuracy + \
ControlH1HanoiDrawHanoiMathDiffAccuracy + \
ControlH2StroopMathStroopHanoiDiffAccuracy + \
ControlH2MathStroopMathHanoiDiffAccuracy


# Inter Conditions Path Errors Stats
InterConditionsPathErrorsStats = statisticize.ttest(SupportivePathErrorsH1H2,
                                              AlternativePathErrorsH1H2, paired=False)#, alternative="greater")

meanSupportivePathErrorsH1H2 = mean(SupportivePathErrorsH1H2)
pstdevSupportivePathErrorsH1H2 = pstdev(SupportivePathErrorsH1H2)

meanAlternativePathErrorsH1H2 = mean(AlternativePathErrorsH1H2)
pstdevAlternativePathErrorsH1H2 = pstdev(AlternativePathErrorsH1H2)

differenceBtwConditionsPathErrors = meanSupportivePathErrorsH1H2 - meanAlternativePathErrorsH1H2
fractionalDifferenceBtwConditionsPathErrors = differenceBtwConditionsPathErrors

InterConditionsPathErrorsStats.insert(0, "Name", "Inter Conditions Path Errors Stats")
InterConditionsPathErrorsStats.insert(9, "Mean Sample 1 (Seconds)", meanSupportivePathErrorsH1H2)
InterConditionsPathErrorsStats.insert(10, "SD Sample 1 (Seconds)", pstdevSupportivePathErrorsH1H2)
InterConditionsPathErrorsStats.insert(11, "Mean Sample 2 (Seconds)", meanAlternativePathErrorsH1H2)
InterConditionsPathErrorsStats.insert(12, "SD Sample 2 (Seconds)", pstdevAlternativePathErrorsH1H2)
InterConditionsPathErrorsStats.insert(13, "Diff In Seconds", differenceBtwConditionsPathErrors)
InterConditionsPathErrorsStats.insert(14, "Fraction In Seconds", fractionalDifferenceBtwConditionsPathErrors)


# Inter Conditions ToH Optimalities Stats
InterConditionsToHOptimalitiesStats = statisticize.ttest(SupportiveToHOptimalitiesH1H2,
                                              AlternativeToHOptimalitiesH1H2, paired=False)#, alternative="greater")

meanSupportiveToHOptimalitiesH1H2 = mean(SupportiveToHOptimalitiesH1H2)
pstdevSupportiveToHOptimalitiesH1H2 = pstdev(SupportiveToHOptimalitiesH1H2)

meanAlternativeToHOptimalitiesH1H2 = mean(AlternativeToHOptimalitiesH1H2)
pstdevAlternativeToHOptimalitiesH1H2 = pstdev(AlternativeToHOptimalitiesH1H2)

differenceBtwConditionsToHOptimalities = meanSupportiveToHOptimalitiesH1H2 - meanAlternativeToHOptimalitiesH1H2
fractionalDifferenceBtwConditionsToHOptimalities = differenceBtwConditionsToHOptimalities

InterConditionsToHOptimalitiesStats.insert(0, "Name", "Inter Conditions ToH Optimalities Stats")
InterConditionsToHOptimalitiesStats.insert(9, "Mean Sample 1 (Seconds)", meanSupportiveToHOptimalitiesH1H2)
InterConditionsToHOptimalitiesStats.insert(10, "SD Sample 1 (Seconds)", pstdevSupportiveToHOptimalitiesH1H2)
InterConditionsToHOptimalitiesStats.insert(11, "Mean Sample 2 (Seconds)", meanAlternativeToHOptimalitiesH1H2)
InterConditionsToHOptimalitiesStats.insert(12, "SD Sample 2 (Seconds)", pstdevAlternativeToHOptimalitiesH1H2)
InterConditionsToHOptimalitiesStats.insert(13, "Diff In Seconds", differenceBtwConditionsToHOptimalities)
InterConditionsToHOptimalitiesStats.insert(14, "Fraction In Seconds", fractionalDifferenceBtwConditionsToHOptimalities)




SupportivePathErrorsH1 = \
ExpH1DrawHanoiDrawStroopDiffAccuracy + \
ExpH1DrawHanoiDrawMathDiffAccuracy

AlternativePathErrorsH1 = \
ControlH1DrawHanoiDrawStroopDiffAccuracy + \
ControlH1DrawHanoiDrawMathDiffAccuracy

SupportiveToHOptimalitiesH1 = \
ExpH1HanoiDrawHanoiStroopDiffAccuracy + \
ExpH1HanoiDrawHanoiMathDiffAccuracy

AlternativeToHOptimalitiesH1 = \
ControlH1HanoiDrawHanoiStroopDiffAccuracy + \
ControlH1HanoiDrawHanoiMathDiffAccuracy

# Path Tracing Error in Supportive (H1) vs Path Tracing Error in Alternative (H1) 
InterventionalH1andComparisonH1ErrorPath = statisticize.ttest(SupportivePathErrorsH1,
                                              AlternativePathErrorsH1, paired=False, alternative="greater")

meanIntraHypothesisOneIntraExperimentationError = mean(SupportivePathErrorsH1)
pstdevIntraHypothesisOneIntraExperimentationError = pstdev(SupportivePathErrorsH1)

meanIntraHypothesisOneIntraControlConditionError = mean(AlternativePathErrorsH1)
pstdevIntraHypothesisOneIntraControlConditionError = pstdev(AlternativePathErrorsH1)

differenceBtwH1EXPandH1CONError = meanIntraHypothesisOneIntraExperimentationError - meanIntraHypothesisOneIntraControlConditionError
fractionaldifferenceBtwH1EXPandH1CONError = differenceBtwH1EXPandH1CONError

InterventionalH1andComparisonH1ErrorPath.insert(0, "Name", "Path Errors in Supportive vs Alternative (H1)")
InterventionalH1andComparisonH1ErrorPath.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationError)
InterventionalH1andComparisonH1ErrorPath.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationError)
InterventionalH1andComparisonH1ErrorPath.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisOneIntraControlConditionError)
InterventionalH1andComparisonH1ErrorPath.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisOneIntraControlConditionError)
InterventionalH1andComparisonH1ErrorPath.insert(13, "Diff In Seconds", differenceBtwH1EXPandH1CONError)
InterventionalH1andComparisonH1ErrorPath.insert(14, "Fraction In Seconds", fractionaldifferenceBtwH1EXPandH1CONError)

# Tower of Hanoi Optimality in Supportive (H1) vs Tower of Hanoi Optimality in Alternative (H1)
InterventionalH1andComparisonH1OptimalityToH = statisticize.ttest(SupportiveToHOptimalitiesH1,
                                              AlternativeToHOptimalitiesH1, paired=False, alternative="greater")

meanIntraHypothesisOneIntraExperimentationOptimality = mean(SupportiveToHOptimalitiesH1)
pstdevIntraHypothesisOneIntraExperimentationOptimality = pstdev(SupportiveToHOptimalitiesH1)

meanIntraHypothesisOneIntraControlConditionOptimality = mean(AlternativeToHOptimalitiesH1)
pstdevIntraHypothesisOneIntraControlConditionOptimality = pstdev(AlternativeToHOptimalitiesH1)

differenceBtwH1EXPandH1CONOptimality = meanIntraHypothesisOneIntraExperimentationOptimality - meanIntraHypothesisOneIntraControlConditionOptimality
fractionaldifferenceBtwH1EXPandH1CONOptimality = differenceBtwH1EXPandH1CONOptimality

InterventionalH1andComparisonH1OptimalityToH.insert(0, "Name", "ToH Optimality in Supportive vs Alternative (H1)")
InterventionalH1andComparisonH1OptimalityToH.insert(9, "Mean Sample 1 (Seconds)", meanIntraHypothesisOneIntraExperimentationOptimality)
InterventionalH1andComparisonH1OptimalityToH.insert(10, "SD Sample 1 (Seconds)", pstdevIntraHypothesisOneIntraExperimentationOptimality)
InterventionalH1andComparisonH1OptimalityToH.insert(11, "Mean Sample 2 (Seconds)", meanIntraHypothesisOneIntraControlConditionOptimality)
InterventionalH1andComparisonH1OptimalityToH.insert(12, "SD Sample 2 (Seconds)", pstdevIntraHypothesisOneIntraControlConditionOptimality)
InterventionalH1andComparisonH1OptimalityToH.insert(13, "Diff In Seconds", differenceBtwH1EXPandH1CONOptimality)
InterventionalH1andComparisonH1OptimalityToH.insert(14, "Fraction In Seconds", fractionaldifferenceBtwH1EXPandH1CONOptimality)





# Left Side (Experimental Intervention) Top left quadrant + Bottom left quadrant
# Top left quadrant interruptions
experimentalInterventionAttentions.append(ExpH1DrawHanoiDrawStroopAttentions)
experimentalInterventionAttentions.append(ExpH1DrawHanoiDrawMathAttentions)
experimentalInterventionAttentions.append(ExpH1HanoiDrawHanoiStroopAttentions)
experimentalInterventionAttentions.append(ExpH1HanoiDrawHanoiMathAttentions)
# Bottom left quadrant interruptions
experimentalInterventionAttentions.append(ExpH2StroopMathStroopDrawAttentions)
experimentalInterventionAttentions.append(ExpH2MathStroopMathDrawAttentions)
experimentalInterventionAttentions.append(ExpH2StroopMathStroopHanoiAttentions)
experimentalInterventionAttentions.append(ExpH2MathStroopMathHanoiAttentions)

averageExperimentalInterventionAttentions = [sum(allExpIntervention) / len(experimentalInterventionAttentions) for
                         allExpIntervention in zip(*experimentalInterventionAttentions)]

first8averageExperimentalInterventionAttentionsAssessment = averageExperimentalInterventionAttentions[:8]

last8averageExperimentalInterventionAttentionsTesting = averageExperimentalInterventionAttentions[16:]

# Plotting of IntraCondition Interruption Stats
plotTitle = 'Assessment vs. Testing Phases in the Experimental Intervention'
yAxisLabel = 'Average Interruption Lag Times (Seconds)'
filenameForPlots = "AvTPhasesInterruptionsComparisonExperimental"
xAxisText = " Lag Times within Each Phase (Averages)"
intraConditionPlotter(first8averageExperimentalInterventionAttentionsAssessment,
                      first8averageExperimentalInterventionAttentionsAssessment,
                      last8averageExperimentalInterventionAttentionsTesting,
                      last8averageExperimentalInterventionAttentionsTesting,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      ExperimentalInterventionInterruptionLags,
                      xAxisText)
# --------------------------------------------------------------------------------
# Top left quadrant resumptions
experimentalInterventionResumptions.append(ExpH1DrawHanoiDrawStroopResumptions)
experimentalInterventionResumptions.append(ExpH1DrawHanoiDrawMathResumptions)
experimentalInterventionResumptions.append(ExpH1HanoiDrawHanoiStroopResumptions)
experimentalInterventionResumptions.append(ExpH1HanoiDrawHanoiMathResumptions)
# Bottom left quadrant resumptions
experimentalInterventionResumptions.append(ExpH2StroopMathStroopDrawResumptions)
experimentalInterventionResumptions.append(ExpH2MathStroopMathDrawResumptions)
experimentalInterventionResumptions.append(ExpH2StroopMathStroopHanoiResumptions)
experimentalInterventionResumptions.append(ExpH2MathStroopMathHanoiResumptions)

averageExperimentalInterventionResumptions = [sum(allExpIntervention) / len(experimentalInterventionResumptions) for
                         allExpIntervention in zip(*experimentalInterventionResumptions)]

first8averageExperimentalInterventionResumptionsAssessment = averageExperimentalInterventionResumptions[:8]

last8averageExperimentalInterventionResumptionsTesting = averageExperimentalInterventionResumptions[16:]

# Plotting of IntraCondition Resumption Stats
plotTitle = 'Assessment vs. Testing Phases in the Experimental Intervention'
yAxisLabel = 'Average Resumption Lag  Times (Seconds)'
filenameForPlots = "AvTPhasesResumptionsComparisonExperimental"
xAxisText = " Lag Times within Each Phase (Averages)"
intraConditionPlotter(first8averageExperimentalInterventionResumptionsAssessment,
                      first8averageExperimentalInterventionResumptionsAssessment,
                      last8averageExperimentalInterventionResumptionsTesting,
                      last8averageExperimentalInterventionResumptionsTesting,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      ExperimentalInterventionResumptionLags,
                      xAxisText)
# --------------------------------------------------------------------------------
# Top left quadrant interruptions
experimentalInterventionAccuracies.append(ExpH1DrawHanoiDrawStroopAccuracies)
experimentalInterventionAccuracies.append(ExpH1DrawHanoiDrawMathAccuracies)
experimentalInterventionAccuracies.append(ExpH1HanoiDrawHanoiStroopAccuracies)
experimentalInterventionAccuracies.append(ExpH1HanoiDrawHanoiMathAccuracies)
# Bottom left quadrant interruptions
experimentalInterventionAccuracies.append(ExpH2StroopMathStroopDrawAccuracies)
experimentalInterventionAccuracies.append(ExpH2MathStroopMathDrawAccuracies)
experimentalInterventionAccuracies.append(ExpH2StroopMathStroopHanoiAccuracies)
experimentalInterventionAccuracies.append(ExpH2MathStroopMathHanoiAccuracies)

averageExperimentalInterventionAccuracies = [sum(allExpIntervention) / len(experimentalInterventionAccuracies) for
                         allExpIntervention in zip(*experimentalInterventionAccuracies)]

first16averageExperimentalInterventionAccuraciesAssessment = averageExperimentalInterventionAccuracies[:16]

last16averageExperimentalInterventionAccuraciesTesting = averageExperimentalInterventionAccuracies[16:]

# Plotting of IntraCondition Accuracies Stats
plotTitle = 'Assessment vs. Testing Phases in the Experimental Intervention'
yAxisLabel = 'Average Error Reductions'
filenameForPlots = "AvTPhasesAccuraciesComparisonExperimental"
xAxisText = " Reductions of Errors within Each Phase (Averages)"
intraConditionPlotter(first16averageExperimentalInterventionAccuraciesAssessment,
                      first16averageExperimentalInterventionAccuraciesAssessment,
                      last16averageExperimentalInterventionAccuraciesTesting,
                      last16averageExperimentalInterventionAccuraciesTesting,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      ExperimentalInterventionAccuracies,
                      xAxisText)
# --------------------------------------------------------------------------------
# Top left quadrant resumptions
experimentalInterventionSpeeds.append(ExpH1DrawHanoiDrawStroopSpeeds)
experimentalInterventionSpeeds.append(ExpH1DrawHanoiDrawMathSpeeds)
experimentalInterventionSpeeds.append(ExpH1HanoiDrawHanoiStroopSpeeds)
experimentalInterventionSpeeds.append(ExpH1HanoiDrawHanoiMathSpeeds)
# Bottom left quadrant resumptions
experimentalInterventionSpeeds.append(ExpH2StroopMathStroopDrawSpeeds)
experimentalInterventionSpeeds.append(ExpH2MathStroopMathDrawSpeeds)
experimentalInterventionSpeeds.append(ExpH2StroopMathStroopHanoiSpeeds)
experimentalInterventionSpeeds.append(ExpH2MathStroopMathHanoiSpeeds)

averageExperimentalInterventionSpeeds = [sum(allExpIntervention) / len(experimentalInterventionSpeeds) for
                         allExpIntervention in zip(*experimentalInterventionSpeeds)]

first16averageExperimentalInterventionSpeedsAssessment = averageExperimentalInterventionSpeeds[:16]

last16averageExperimentalInterventionSpeedsTesting = averageExperimentalInterventionSpeeds[16:]

# Plotting of IntraCondition Speed Stats
plotTitle = 'Assessment vs. Testing Phases in the Experimental Intervention'
# yAxisLabel = 'Average Reduction of Time Spent Performing Tasks'
yAxisLabel = 'Average Completion Time Saved for Performing Tasks'
filenameForPlots = "AvTPhasesSpeedComparisonExperimental"
xAxisText = " Reductions in Time within Each Phase (Averages)"
intraConditionPlotter(first16averageExperimentalInterventionSpeedsAssessment,
                      first16averageExperimentalInterventionSpeedsAssessment,
                      last16averageExperimentalInterventionSpeedsTesting,
                      last16averageExperimentalInterventionSpeedsTesting,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      ExperimentalInterventionSpeed,
                      xAxisText)
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Right Side (Control Comparison) Top right quadrant + Bottom right quadrant
# Top right quadrant interruptions
controlComparisonAttentions.append(ControlH1DrawHanoiDrawStroopAttentions)
controlComparisonAttentions.append(ControlH1DrawHanoiDrawMathAttentions)
controlComparisonAttentions.append(ControlH1HanoiDrawHanoiStroopAttentions)
controlComparisonAttentions.append(ControlH1HanoiDrawHanoiMathAttentions)
# Bottom right quadrant interruptions
controlComparisonAttentions.append(ControlH2StroopMathStroopDrawAttentions)
controlComparisonAttentions.append(ControlH2MathStroopMathDrawAttentions)
controlComparisonAttentions.append(ControlH2StroopMathStroopHanoiAttentions)
controlComparisonAttentions.append(ControlH2MathStroopMathHanoiAttentions)

averageControlComparisonAttentions = [sum(allExpIntervention) / len(controlComparisonAttentions) for
                         allExpIntervention in zip(*controlComparisonAttentions)]

first8averageControlComparisonAttentionsAssessment = averageControlComparisonAttentions[:8]

last8averageControlComparisonAttentionsTesting = averageControlComparisonAttentions[16:]

# Plotting of IntraCondition Interruption Stats
plotTitle = 'Assessment vs. Testing Phases in the Control Comparison'
yAxisLabel = 'Average Interruption Lag Times (Seconds)'
filenameForPlots = "AvTPhasesInterruptionsComparisonControl"
xAxisText = " Lag Times within Each Phase (Averages)"
intraConditionPlotter(first8averageControlComparisonAttentionsAssessment,
                      first8averageControlComparisonAttentionsAssessment,
                      last8averageControlComparisonAttentionsTesting,
                      last8averageControlComparisonAttentionsTesting,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      ControlComparisonInterruptionLags,
                      xAxisText)
# --------------------------------------------------------------------------------
# Top right quadrant resumptions
controlComparisonResumptions.append(ControlH1DrawHanoiDrawStroopResumptions)
controlComparisonResumptions.append(ControlH1DrawHanoiDrawMathResumptions)
controlComparisonResumptions.append(ControlH1HanoiDrawHanoiStroopResumptions)
controlComparisonResumptions.append(ControlH1HanoiDrawHanoiMathResumptions)
# Bottom right quadrant resumptions
controlComparisonResumptions.append(ControlH2StroopMathStroopDrawResumptions)
controlComparisonResumptions.append(ControlH2MathStroopMathDrawResumptions)
controlComparisonResumptions.append(ControlH2StroopMathStroopHanoiResumptions)
controlComparisonResumptions.append(ControlH2MathStroopMathHanoiResumptions)

averageControlComparisonResumptions = [sum(allExpIntervention) / len(controlComparisonResumptions) for
                         allExpIntervention in zip(*controlComparisonResumptions)]

first8averageControlComparisonResumptionsAssessment = averageControlComparisonResumptions[:8]

last8averageControlComparisonResumptionsTesting = averageControlComparisonResumptions[16:]

# Plotting of IntraCondition Resumption Stats
plotTitle = 'Assessment vs. Testing Phases in the Control Comparison'
yAxisLabel = 'Average Resumption Lag  Times (Seconds)'
filenameForPlots = "AvTPhasesResumptionsComparisonControl"
xAxisText = " Lag Times within Each Phase (Averages)"
intraConditionPlotter(first8averageControlComparisonResumptionsAssessment,
                      first8averageControlComparisonResumptionsAssessment,
                      last8averageControlComparisonResumptionsTesting,
                      last8averageControlComparisonResumptionsTesting,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      ControlComparisonResumptionLags,
                      xAxisText)
# --------------------------------------------------------------------------------
# Top right quadrant interruptions
controlComparisonAccuracies.append(ControlH1DrawHanoiDrawStroopAccuracies)
controlComparisonAccuracies.append(ControlH1DrawHanoiDrawMathAccuracies)
controlComparisonAccuracies.append(ControlH1HanoiDrawHanoiStroopAccuracies)
controlComparisonAccuracies.append(ControlH1HanoiDrawHanoiMathAccuracies)
# Bottom right quadrant interruptions
controlComparisonAccuracies.append(ControlH2StroopMathStroopDrawAccuracies)
controlComparisonAccuracies.append(ControlH2MathStroopMathDrawAccuracies)
controlComparisonAccuracies.append(ControlH2StroopMathStroopHanoiAccuracies)
controlComparisonAccuracies.append(ControlH2MathStroopMathHanoiAccuracies)

averageControlComparisonAccuracies = [sum(allExpIntervention) / len(controlComparisonAccuracies) for
                         allExpIntervention in zip(*controlComparisonAccuracies)]

first16averageControlComparisonAccuraciesAssessment = averageControlComparisonAccuracies[:16]

last16averageControlComparisonAccuraciesTesting = averageControlComparisonAccuracies[16:]

# Plotting of IntraCondition Accuracies Stats
plotTitle = 'Assessment vs. Testing Phases in the Control Comparison'
yAxisLabel = 'Average Error Reductions'
filenameForPlots = "AvTPhasesAccuraciesComparisonControl"
xAxisText = " Reductions of Errors within Each Phase (Averages)"
intraConditionPlotter(first16averageControlComparisonAccuraciesAssessment,
                      first16averageControlComparisonAccuraciesAssessment,
                      last16averageControlComparisonAccuraciesTesting,
                      last16averageControlComparisonAccuraciesTesting,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      ControlComparisonAccuracies,
                      xAxisText)
# --------------------------------------------------------------------------------
# Top right quadrant resumptions
controlComparisonSpeeds.append(ControlH1DrawHanoiDrawStroopSpeeds)
controlComparisonSpeeds.append(ControlH1DrawHanoiDrawMathSpeeds)
controlComparisonSpeeds.append(ControlH1HanoiDrawHanoiStroopSpeeds)
controlComparisonSpeeds.append(ControlH1HanoiDrawHanoiMathSpeeds)
# Bottom right quadrant resumptions
controlComparisonSpeeds.append(ControlH2StroopMathStroopDrawSpeeds)
controlComparisonSpeeds.append(ControlH2MathStroopMathDrawSpeeds)
controlComparisonSpeeds.append(ControlH2StroopMathStroopHanoiSpeeds)
controlComparisonSpeeds.append(ControlH2MathStroopMathHanoiSpeeds)

averageControlComparisonSpeeds = [sum(allExpIntervention) / len(controlComparisonSpeeds) for
                         allExpIntervention in zip(*controlComparisonSpeeds)]

first16averageControlComparisonSpeedsAssessment = averageControlComparisonSpeeds[:16]

last16averageControlComparisonSpeedsTesting = averageControlComparisonSpeeds[16:]

# Plotting of IntraCondition Speed Stats
plotTitle = 'Assessment vs. Testing Phases in the Control Comparison'
yAxisLabel = 'Average Completion Time Saved for Performing Tasks'
filenameForPlots = "AvTPhasesSpeedComparisonControl"
xAxisText = " Reductions in Time within Each Phase (Averages)"
intraConditionPlotter(first16averageControlComparisonSpeedsAssessment,
                      first16averageControlComparisonSpeedsAssessment,
                      last16averageControlComparisonSpeedsTesting,
                      last16averageControlComparisonSpeedsTesting,
                      plotTitle,
                      yAxisLabel,
                      PlotPlace,
                      filenameForPlots,
                      ControlComparisonSpeed,
                      xAxisText)




# Concatenating the DataFrames containing inter-hypothesis' metrics' stats
TopLevelOneStats = pd.concat([InterConditionMetricsStatsHeader,
InterConditionsResumptionStats,
InterConditionsInterruptionStats,
InterConditionsAccuracyStats,
InterConditionsPathErrorsStats,
InterConditionsToHOptimalitiesStats,
InterConditionsSpeedStats,
intraHypothesisInterConditionMetricsStatsHeader,
IntraHypothesisInterConditionsResumptionStats,
IntraHypothesisInterConditionsInterruptionStats,
IntraHypothesisInterConditionsAccuraciesStats,
IntraHypothesisInterConditionsSpeedsStats,
intraHypothesisMetricsStatsExpHeader,
IntraHypothesisIntraExperimentationResumptionStats,
IntraHypothesisIntraExperimentationInterruptionStats,
IntraHypothesisIntraExperimentationAccuraciesStats,
IntraHypothesisIntraExperimentationSpeedsStats,
intraHypothesisMetricsStatsControlHeader,
IntraHypothesisIntraControlResumptionStats,
IntraHypothesisIntraControlInterruptionStats,
IntraHypothesisIntraControlAccuraciesStats,
IntraHypothesisIntraControlSpeedsStats,
AssessTestingPathTracingH1ExperimentalHeader,
PathTracingExperimentalInterventionResumptionLagsH1,
PathTracingExperimentalInterventionInterruptionLagsH1,
PathTracingExperimentalInterventionAccuraciesH1,
PathTracingExperimentalInterventionSpeedH1,
AssessTestingInToHH1ExperimentalHeader,
ToHExperimentalInterventionResumptionLagsH1,
ToHExperimentalInterventionInterruptionLagsH1,
ToHExperimentalInterventionAccuraciesH1,
ToHExperimentalInterventionSpeedH1,
AssessTestingPathTracingH1ControlHeader,
PathTracingControlComparisonResumptionLagsH1,
PathTracingControlComparisonInterruptionLagsH1,
PathTracingControlComparisonAccuraciesH1,
PathTracingControlComparisonSpeedH1,
AssessTestingToHH1ControlHeader,
ToHControlComparisonResumptionLagsH1,
ToHControlComparisonInterruptionLagsH1,
ToHControlComparisonAccuraciesH1,
ToHControlComparisonSpeedH1,
AssessTestingPathTracingH2ExperimentalHeader,
PathTracingExperimentalInterventionResumptionLagsH2,
PathTracingExperimentalInterventionInterruptionLagsH2,
PathTracingExperimentalInterventionAccuraciesH2,
PathTracingExperimentalInterventionSpeedH2,
AssessTestingInToHH2ExperimentalHeader,
ToHExperimentalInterventionResumptionLagsH2,
ToHExperimentalInterventionInterruptionLagsH2,
ToHExperimentalInterventionAccuraciesH2,
ToHExperimentalInterventionSpeedH2,
AssessTestingPathTracingH2ControlHeader,
PathTracingControlComparisonResumptionLagsH2,
PathTracingControlComparisonInterruptionLagsH2,
PathTracingControlComparisonAccuraciesH2,
PathTracingControlComparisonSpeedH2,
AssessTestingToHH2ControlHeader,
ToHControlComparisonResumptionLagsH2,
ToHControlComparisonInterruptionLagsH2,
ToHControlComparisonAccuraciesH2,
ToHControlComparisonSpeedH2,
TaskTypeComparisonsforPerformanceMetricsHeader,
TaskTypeComparisonResume,
TaskTypeComparisonAttend,
TaskTypeComparisonAccuracy,
TaskTypeComparisonSpeed,
TaskTypeComparisonsEXPforPerformanceMetricsHeader,
TaskTypeComparisonResumeEXP,
TaskTypeComparisonAttendEXP,
TaskTypeComparisonAccuracyEXP,
TaskTypeComparisonSpeedEXP,
TaskTypeComparisonsCONforPerformanceMetricsHeader,
TaskTypeComparisonResumeCON,
TaskTypeComparisonAttendCON,
TaskTypeComparisonAccuracyCON,
TaskTypeComparisonSpeedCON,
AssessTestingPathTracingExperimentalHeader,
PathTracingExperimentalInterventionResumptionLags,
PathTracingExperimentalInterventionInterruptionLags,
PathTracingExperimentalInterventionAccuracies,
PathTracingExperimentalInterventionSpeed,
AssessTestingInToHExperimentalHeader,
ToHExperimentalInterventionResumptionLags,
ToHExperimentalInterventionInterruptionLags,
ToHExperimentalInterventionAccuracies,
ToHExperimentalInterventionSpeed,
AssessTestingPathTracingControlHeader,
PathTracingControlComparisonResumptionLags,
PathTracingControlComparisonInterruptionLags,
PathTracingControlComparisonAccuracies,
PathTracingControlComparisonSpeed,
AssessTestingToHControlHeader,
ToHControlComparisonResumptionLags,
ToHControlComparisonInterruptionLags,
ToHControlComparisonAccuracies,
ToHControlComparisonSpeed,
MetricsH1ExperimentalControlHeader,
InterventionalH1andComparisonH1Resume,
InterventionalH1andComparisonH1Attend,
InterventionalH1andComparisonH1Accuracy,
InterventionalH1andComparisonH1ErrorPath,
InterventionalH1andComparisonH1OptimalityToH,
InterventionalH1andComparisonH1Speed,
MetricsH2ExperimentalControlHeader,
InterventionalH2andComparisonH2Resume,
InterventionalH2andComparisonH2Attend,
InterventionalH2andComparisonH2Accuracy,
InterventionalH2andComparisonH2Speed,
PerformanceMetricH1EXPplusCONHeader,
RLH1EXPplusCONStats,
ILH1EXPplusCONStats,
ACH1EXPplusCONStats,
errorRatesBTWAssessTestH1,
optimalityBTWAssessTestH1,
SPH1EXPplusCONStats,
PerformanceMetricH2EXPplusCONHeader,
RLH2EXPplusCONStats,
ILH2EXPplusCONStats,
ACH2EXPplusCONStats,
errorRatesBTWAssessTestH2,
optimalityBTWAssessTestH2,
SPH2EXPplusCONStats,
AssessTestingInH1ExperimentalHeader,
H1ExperimentalInterventionResumptionLags,
H1ExperimentalInterventionInterruptionLags,
H1ExperimentalInterventionAccuracies,
H1ExperimentalInterventionSpeed,
AssessTestingInH2ExperimentalHeader,
H2ExperimentalInterventionResumptionLags,
H2ExperimentalInterventionInterruptionLags,
H2ExperimentalInterventionAccuracies,
H2ExperimentalInterventionSpeed,
AssessTestingInH1ControlHeader,
H1ControlComparisonResumptionLags,
H1ControlComparisonInterruptionLags,
H1ControlComparisonAccuracies,
H1ControlComparisonSpeed,
AssessTestingInH2ControlHeader,
H2ControlComparisonResumptionLags,
H2ControlComparisonInterruptionLags,
H2ControlComparisonAccuracies,
H2ControlComparisonSpeed,
AssessTestingInExperimentalHeader,
ExperimentalInterventionResumptionLags,
ExperimentalInterventionInterruptionLags,
ExperimentalInterventionAccuracies,
ExperimentalInterventionSpeed,
AssessTestingInControlHeader,
ControlComparisonResumptionLags,
ControlComparisonInterruptionLags,
ControlComparisonAccuracies,
ControlComparisonSpeed,
PerformanceMetricSPBothHypothesesEXPplusCONStatsEXPplusCONHeader,
RLBothHypothesesEXPplusCONStats,
ILBothHypothesesEXPplusCONStats,
ACBothHypothesesEXPplusCONStats,
accuracyBothHypothesesEXPplusCONStats,
optimalityBothHypothesesEXPplusCONStats,
SPBothHypothesesEXPplusCONStats
                                 ])
filenameForStats = "summarizationStats"
TopLevelOneStats.to_csv('../DataResults/Stats/SummarizingStats/' + filenameForStats + '.csv')




# Finding fewest number of moves per hanoi task in Assessment
fewestHanoiMovesPerTaskAssessment = [min(allParticipantsSeriesOfMovesPerTask) for
                           allParticipantsSeriesOfMovesPerTask in
                           zip(*numberOfMovesToCompleteHanoiStackedAssessment)]

avgfewestHanoiMovesPerTaskAssessment = [
    sum(allParticipantsSeriesOfMovesPerTask) / len(allParticipantsSeriesOfMovesPerTask)
    for
    allParticipantsSeriesOfMovesPerTask in
    zip(*numberOfMovesToCompleteHanoiStackedAssessment)]

highestHanoiMovesPerTask = [max(allParticipantsSeriesOfMovesPerTask) for
                            allParticipantsSeriesOfMovesPerTask in
                            zip(*numberOfMovesToCompleteHanoiStackedAssessment)]
# accuracyInAssessmentSum = [n / b for n, b in
#                            zip(numberOfMovesToCompleteHanoiAssessment, fewestHanoiMovesPerTaskAssessment)]
# accuracyInAssessmentSum = sum(accuracyInAssessmentSum)

baseVariable = "hanoiTask"
taskSequenceContainer = []
taskSequenceValueContainer = []
dictHanoiTasksMovesAssessment = {}
for i in range(0, len(fewestHanoiMovesPerTaskAssessment)):
    numberOfMovesToCompleteHanoiAssessmentForDict = fewestHanoiMovesPerTaskAssessment[i]
    taskSequenceValueContainer.append(numberOfMovesToCompleteHanoiAssessmentForDict)
    baseVariable = "hanoiTask#"
    iterantString = str(i)
    baseVariable += iterantString
    taskSequenceContainer.append(baseVariable)
    dictHanoiTasksMovesAssessment.update({taskSequenceContainer[i]: taskSequenceValueContainer[i]})

# Finding fewest number of moves per hanoi task in Testing
fewestHanoiMovesPerTaskTesting = [min(allParticipantsSeriesOfMovesPerTask) for
                                  allParticipantsSeriesOfMovesPerTask in
                                  zip(*numberOfMovesToCompleteHanoiStackedTesting)]

avgfewestHanoiMovesPerTaskTesting = [
    sum(allParticipantsSeriesOfMovesPerTask) / len(allParticipantsSeriesOfMovesPerTask) for
    allParticipantsSeriesOfMovesPerTask in zip(*numberOfMovesToCompleteHanoiStackedTesting)]
highestHanoiMovesPerTaskTesting = [max(allParticipantsSeriesOfMovesPerTask) for
                                   allParticipantsSeriesOfMovesPerTask in
                                   zip(*numberOfMovesToCompleteHanoiStackedTesting)]
# accuracyInTestingSum = [n / b for n, b in
#                         zip(numberOfMovesToCompleteHanoiTesting, fewestHanoiMovesPerTaskTesting)]
# accuracyInTestingSum = sum(accuracyInTestingSum)

baseVariableTesting = "hanoiTask"
taskSequenceContainerTesting = []
taskSequenceValueContainerTesting = []
dictHanoiTasksMovesTesting = {}
for i in range(0, len(fewestHanoiMovesPerTaskTesting)):
    numberOfMovesToCompleteHanoiTestingForDict = fewestHanoiMovesPerTaskTesting[i]
    taskSequenceValueContainerTesting.append(numberOfMovesToCompleteHanoiTestingForDict)
    baseVariableTesting = "hanoiTaskTesting#"
    iteratorString = str(i)
    baseVariableTesting += iteratorString
    taskSequenceContainerTesting.append(baseVariableTesting)
    dictHanoiTasksMovesTesting.update({taskSequenceContainerTesting[i]: taskSequenceValueContainerTesting[i]})



columnTitles = {
    "PID": id_arr, 
    "Starting Interruption": starting_interruption_arr,
    "Starting Task": starting_task_arr,
    "Hypotheses": conditions_arr,
    "Control": control_arr,
    
    "d_age": d_age,
    "d_gender": d_gender,
    "d_education": d_education,

    "d_asd": d_asd,
    "d_colorblind": d_colorblind,
    "d_hearingimpaired": d_hearingimpaired,
    "d_adhd": d_adhd,
    "d_prefernottosay": d_prefernottosay,
    "d_none": d_none,

    "a_p_e_task": a_p_e_task,
    "a_p_e_effort": a_p_e_effort,
    "a_p_e_confidence": a_p_e_confidence,

    "a_i_e_task": a_i_e_task,
    "a_i_e_effort": a_i_e_effort,
    "a_i_e_confidence": a_i_e_confidence,

    "tr_p_e_task": tr_p_e_task,
    "tr_p_e_effort": tr_p_e_effort,
    "tr_p_e_confidence": tr_p_e_confidence,

    "tr_i_e_task": tr_i_e_task,
    "tr_i_e_effort": tr_i_e_effort,
    "tr_i_e_confidence": tr_i_e_confidence,

    "te_p_e_task": te_p_e_task,
    "te_p_e_effort": te_p_e_effort,
    "te_p_e_confidence": te_p_e_confidence,

    # Unpopulated lists embedded in the dictionary, so commented out
    "te_i_e_task": te_i_e_task,
    "te_i_e_effort": te_i_e_effort,
    "te_i_e_confidence": te_i_e_confidence,

    "a_i_name": a_i_name, 
    "a_i_count": a_i_count,
    "a_i_percentage": a_i_percentage,
    "a_i_time": a_i_time,
    "a_i_times": a_i_times,
    
    "a_p_name": a_p_name,
    "a_p_count": a_p_count,
    "a_p_correctness": a_p_correctness,
    "a_p_time": a_p_time,
    "a_p_times": a_p_times,
    "a_p_percentage": a_p_percentage,
    "a_p_percentage100": a_p_percentage100,
    "a_p_resumption": a_p_resumption, 
    "a_p_resumptions": a_p_resumptions,
    "a_p_interruptions": a_p_interruptions, 
    "a_p_movestotal": a_p_movestotal,

    "tr_i_name": tr_i_name, 
    "tr_i_count": tr_i_count,
    "tr_i_percentage": tr_i_percentage,
    "tr_i_time": tr_i_time,
    "tr_i_times": tr_i_times,
    
    "tr_p_name": tr_p_name,
    "tr_p_count": tr_p_count,
    "tr_p_correctness": tr_p_correctness,
    "tr_p_time": tr_p_time,
    "tr_p_times": tr_p_times,
    "tr_p_percentage": tr_p_percentage,
    "tr_p_percentage100": tr_p_percentage100,
    "tr_p_resumption": tr_p_resumption, 
    "tr_p_resumptions": tr_p_resumptions,
    "tr_p_interruptions": tr_p_interruptions, 
    "tr_p_movestotal": tr_p_movestotal,

    "te_i_name": te_i_name, 
    "te_i_count": te_i_count,
    "te_i_percentage": te_i_percentage,
    "te_i_time": te_i_time,
    "te_i_times": te_i_times,
    
    "te_p_name": te_p_name,
    "te_p_count": te_p_count,
    "te_p_correctness": te_p_correctness,
    "te_p_time": te_p_time,
    "te_p_times": te_p_times,
    "te_p_percentage": te_p_percentage,
    "te_p_percentage100": te_p_percentage100,
    "te_p_resumption": te_p_resumption, 
    "te_p_resumptions": te_p_resumptions,
    "te_p_interruptions": te_p_interruptions, 
    "te_p_movestotal": te_p_movestotal 
    }

dataframe = pd.DataFrame(columnTitles)
dataframe.to_csv('../DataResults/results.csv')
print("METRICS EXPORTED SUCCESSFULLY")