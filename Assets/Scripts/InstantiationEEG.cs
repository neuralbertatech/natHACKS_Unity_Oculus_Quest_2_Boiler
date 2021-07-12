using System;
using UnityEngine;

using Accord.Math;
using brainflow;


public class InstantiationEEG : MonoBehaviour
{
    
    // ----------------------------------------------- //
    // Brain data elements
    // ----------------------------------------------- //


    public bool simulation; // if 1 simulates brain data, if 0 from board
    private double[] filtered;

    private BoardShim boardShim = null;
    private BrainFlowInputParams input_params = null;
    private MLModel concentration = null;
    private static int board_id = 0;
    private int sampling_rate = 0;
    private int[] eeg_channels = null;
    private int[] accelChannels = null;


    public void Start()
    {

        // ----------------------------------------------- //
        // Brain data elements
        // ----------------------------------------------- //

        try
        {
            input_params = new BrainFlowInputParams();

            BoardShim.set_log_file("brainflow_log.txt");
            BoardShim.enable_dev_board_logger();

            if (simulation)
            {
                board_id = (int)BoardIds.SYNTHETIC_BOARD;
            } else
            {
                input_params.serial_port = "COM3";
                board_id = (int)BoardIds.CYTON_DAISY_BOARD;
                accelChannels = BoardShim.get_accel_channels(board_id);
            }

            boardShim = new BoardShim(board_id, input_params);
            boardShim.prepare_session();
            boardShim.start_stream(450000, "file://brainflow_data.csv:w");
            BrainFlowModelParams concentration_params = new BrainFlowModelParams((int)BrainFlowMetrics.CONCENTRATION, (int)BrainFlowClassifiers.REGRESSION);
            concentration = new MLModel(concentration_params);
            concentration.prepare();

            sampling_rate = BoardShim.get_sampling_rate(board_id);
            eeg_channels = BoardShim.get_eeg_channels(board_id);
            Debug.Log("Brainflow streaming was started");
        }
        catch (BrainFlowException e)
        {
            Debug.Log(e);
        }
    }

    // Update is called once per frame
    void Update()
    {     
        // ----------------------------------------------- //
        // Brain data elements
        // ----------------------------------------------- //

        if ((boardShim == null) || (concentration == null))
        {
            return;
        }
        
        int number_of_data_points = sampling_rate * 4; // 4 second window is recommended for concentration and relaxation calculations
        
        double[,] unprocessed_data = boardShim.get_current_board_data(number_of_data_points);
        if (unprocessed_data.GetRow(0).Length < number_of_data_points)
        {
            return; // wait for more data
        }

        for (int i = 0; i < eeg_channels.Length; i++)
        {
            filtered = DataFilter.perform_wavelet_denoising (unprocessed_data.GetRow (eeg_channels[i]), "db4", 3);
            // Debug.Log("channel " + eeg_channels[i] + " = " + filtered[i].ToString());
        }

        // prepare feature vector
        Tuple<double[], double[]> bands = DataFilter.get_avg_band_powers (unprocessed_data, eeg_channels, sampling_rate, true);
        
        double[] feature_vector = bands.Item1.Concatenate (bands.Item2);
        Debug.Log("Concentration: " + concentration.predict (feature_vector)); // calc and print concetration level
        
    }

    // ----------------------------------------------- //
    // Brain data functions
    // ----------------------------------------------- //

    void OnDestroy()
    {
        if (boardShim != null)
        {
            try
            {
                boardShim.release_session();
                concentration.release();
            }
            catch (BrainFlowException e)
            {
                Debug.Log(e);
            }
            Debug.Log("Brainflow streaming was stopped");
        }
    }
}

