import numpy as np
from scipy import stats as st
from scipy.stats import norm


def confidence_interval_t(data, confidence=0.95):
    data_array = 1.0 * np.array(data)
    degree_of_freedom = len(data_array) - 1
    sample_mean, sample_standard_error = np.mean(data_array), st.sem(data_array)
    t = st.t.ppf((1 + confidence) / 2., degree_of_freedom)
    margin_of_error_greedy = sample_standard_error * t
    Confidence_Interval = 1.0 * np.array([sample_mean - margin_of_error_greedy, sample_mean + margin_of_error_greedy])
    return sample_mean, Confidence_Interval, margin_of_error_greedy

def confidence_interval_normal(data, confidence=0.95):
    data_array = 1.0 * np.array(data)
    sample_mean, sample_standard_error = np.mean(data_array), st.sem(data_array)
    z = norm().ppf((1 + confidence) / 2.)
    margin_of_error = sample_standard_error * z
    Confidence_Interval = np.array([sample_mean - margin_of_error, sample_mean + margin_of_error])
    return sample_mean, Confidence_Interval, margin_of_error

def confidence_interval_init(data, confidence=0.95):
    data_array = 1.0 * np.array(data)
    dimensions = data_array.shape
    if len(dimensions) > 1:
        rows, columns = dimensions[0], dimensions[1]
        if columns <= 30:
            sample_mean, Confidence_Interval_array, margin_of_error_greedy = confidence_interval_t(data_array[0], confidence=0.95)
            sample_mean_array = 1.0 * np.array(sample_mean)
            margin_of_error_greedy_array = 1.0 * np.array(margin_of_error_greedy)
            for row in range(1,rows):
                sample_mean_new_row, Confidence_Interval_new_row, margin_of_error_greedy_new_row =\
                    confidence_interval_t(data_array[row], confidence=0.95)
                sample_mean_array = np.append(sample_mean_array, sample_mean_new_row)
                Confidence_Interval_array = np.vstack((Confidence_Interval_array, Confidence_Interval_new_row))
                margin_of_error_greedy_array = np.append(margin_of_error_greedy_array, margin_of_error_greedy_new_row)
            return sample_mean_array, Confidence_Interval_array, margin_of_error_greedy_array
        else:
            sample_mean, Confidence_Interval_array, margin_of_error_greedy = \
                confidence_interval_normal(data_array[0], confidence=0.95)
            sample_mean_array = 1.0 * np.array(sample_mean)
            margin_of_error_greedy_array = 1.0 * np.array(margin_of_error_greedy)
            for row in range(1, rows):
                sample_mean_new_row, Confidence_Interval_new_row, margin_of_error_greedy_new_row = \
                    confidence_interval_normal(data_array[row], confidence=0.95)
                sample_mean_array = np.append(sample_mean_array, sample_mean_new_row)
                Confidence_Interval_array = np.vstack((Confidence_Interval_array, Confidence_Interval_new_row))
                margin_of_error_greedy_array = np.append(margin_of_error_greedy_array, margin_of_error_greedy_new_row)
            return sample_mean_array, Confidence_Interval_array, margin_of_error_greedy_array
    else:
        if len(data_array <= 30):
            return confidence_interval_t(data_array, confidence=0.95)
        else:
            return confidence_interval_normal(data_array, confidence=0.95)