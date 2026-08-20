<?xml version="1.0" encoding="UTF-8"?>
<!-- Data Browser config embedded by leak_predict.bob: live trend of the
     causal leak-detection signal, LeakML:LevelRate (d(level_corrected)/dt,
     mm/s). Sustained negative excursions over the ~15-minute detection
     window are what drive LeakML:LeakDetected via the physics gate. -->
<databrowser>
  <title>Level rate — d(level)/dt</title>
  <save_changes>false</save_changes>
  <grid>true</grid>
  <scroll>true</scroll>
  <update_period>1.0</update_period>
  <scroll_step>5</scroll_step>
  <start>-15 minutes</start>
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
      <name>Level rate (mm/s)</name>
      <use_axis_name>true</use_axis_name>
      <use_trace_names>false</use_trace_names>
      <right>false</right>
      <min>-1.0</min>
      <max>1.0</max>
      <grid>true</grid>
      <autoscale>true</autoscale>
      <log_scale>false</log_scale>
    </axis>
  </axes>
  <pvlist>
    <pv>
      <display_name>Level rate</display_name>
      <visible>true</visible>
      <name>CWM-CWS02:WtrC-LeakML:LevelRate</name>
      <axis>0</axis>
      <trace_type>LINE</trace_type>
      <linewidth>2</linewidth>
      <color><red>40</red><green>110</green><blue>200</blue></color>
      <period>0.0</period>
      <scan_period>1.0</scan_period>
      <ring_size>5000</ring_size>
      <waveform_index>0</waveform_index>
      <request>RAW</request>
    </pv>
  </pvlist>
</databrowser>
