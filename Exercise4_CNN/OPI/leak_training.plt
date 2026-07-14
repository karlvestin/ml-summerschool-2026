<?xml version="1.0" encoding="UTF-8"?>
<!-- Data Browser config embedded by leak_training.bob: live trend of the
     model's PREDICTED leak-detected boolean against the simulator's TRUE
     leak rate. FOLLOW-UP: TRUE (Sim:TrueLeak, continuous L/min) and
     PREDICTED (LeakML:LeakDetected, boolean 0/1) are no longer the same
     kind of quantity now that the model is a classifier — the shared
     "Leak flow rate (L/min)" axis below is stale/misleading until
     training_sim.py's Sim:* truth signal is reworked for classification. -->
<databrowser>
  <title>True leak rate vs Predicted leak-detected</title>
  <save_changes>false</save_changes>
  <grid>true</grid>
  <scroll>true</scroll>
  <update_period>1.0</update_period>
  <scroll_step>5</scroll_step>
  <start>-3 minutes</start>
  <end>now</end>
  <archive_rescale>STAGGER</archive_rescale>
  <time_axis>
    <name>Time</name>
    <use_axis_name>false</use_axis_name>
    <use_trace_names>true</use_trace_names>
    <visible>true</visible>
    <grid>false</grid>
  </time_axis>
  <axes>
    <axis>
      <visible>true</visible>
      <name>Leak flow rate (L/min)</name>
      <use_axis_name>true</use_axis_name>
      <use_trace_names>false</use_trace_names>
      <right>false</right>
      <min>-2.0</min>
      <max>14.0</max>
      <grid>true</grid>
      <autoscale>false</autoscale>
      <log_scale>false</log_scale>
    </axis>
  </axes>
  <pvlist>
    <pv>
      <display_name>TRUE leak</display_name>
      <visible>true</visible>
      <name>CWM-CWS02:WtrC-Sim:TrueLeak</name>
      <axis>0</axis>
      <trace_type>STEP</trace_type>
      <linewidth>3</linewidth>
      <color><red>40</red><green>160</green><blue>60</blue></color>
      <period>0.0</period>
      <scan_period>1.0</scan_period>
      <ring_size>5000</ring_size>
      <waveform_index>0</waveform_index>
      <request>RAW</request>
    </pv>
    <pv>
      <display_name>PREDICTED — leak detected</display_name>
      <visible>true</visible>
      <name>CWM-CWS02:WtrC-LeakML:LeakDetected</name>
      <axis>0</axis>
      <trace_type>STEP</trace_type>
      <linewidth>3</linewidth>
      <color><red>200</red><green>40</green><blue>40</blue></color>
      <period>0.0</period>
      <scan_period>1.0</scan_period>
      <ring_size>5000</ring_size>
      <waveform_index>0</waveform_index>
      <request>RAW</request>
    </pv>
  </pvlist>
</databrowser>
