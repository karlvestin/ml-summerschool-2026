<?xml version="1.0" encoding="UTF-8"?>
<!-- Data Browser config embedded by leak_training.bob: ONE consolidated
     trend of all four LIVE (real skid) sensors — base PV names, same as
     leak_predict.bob's consolidated trend (leak_predict_sensors.plt). New
     panel: the training bench previously had no raw live-sensor trend at
     all (only the simulated-sensor mini-plots and the TRUE-vs-PREDICTED
     trend). Lets an operator compare the replay stream against the real
     skid side by side. Three value axes keep degC / mm / % from flattening
     each other:
       axis 0 (left)  — Temperature, degC  — TT-102, TT-110
       axis 1 (right) — Level, mm          — LI-103
       axis 2 (right) — Pump speed, %      — P-101 Speed
     Trace colors match the same convention used everywhere else in this
     project (blue/red/green/purple per sensor). -->
<databrowser>
  <title>Live Sensors — Real Skid PVs</title>
  <save_changes>false</save_changes>
  <grid>true</grid>
  <scroll>true</scroll>
  <update_period>1.0</update_period>
  <scroll_step>5</scroll_step>
  <start>-15 minutes</start>
  <end>now</end>
  <archive_rescale>STAGGER</archive_rescale>
  <show_legend>true</show_legend>
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
      <name>Temperature (degC)</name>
      <use_axis_name>true</use_axis_name>
      <use_trace_names>false</use_trace_names>
      <right>false</right>
      <min>10.0</min>
      <max>40.0</max>
      <grid>true</grid>
      <autoscale>true</autoscale>
      <log_scale>false</log_scale>
    </axis>
    <axis>
      <visible>true</visible>
      <name>Vessel level (mm)</name>
      <use_axis_name>true</use_axis_name>
      <use_trace_names>false</use_trace_names>
      <right>true</right>
      <min>0.0</min>
      <max>300.0</max>
      <grid>false</grid>
      <autoscale>true</autoscale>
      <log_scale>false</log_scale>
    </axis>
    <axis>
      <visible>true</visible>
      <name>Pump speed (%)</name>
      <use_axis_name>true</use_axis_name>
      <use_trace_names>false</use_trace_names>
      <right>true</right>
      <min>0.0</min>
      <max>100.0</max>
      <grid>false</grid>
      <autoscale>true</autoscale>
      <log_scale>false</log_scale>
    </axis>
  </axes>
  <pvlist>
    <pv>
      <display_name>Return temp (TT-102)</display_name>
      <visible>true</visible>
      <name>CWM-CWS02:WtrC-TT-102:MeasValue</name>
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
    <pv>
      <display_name>Supply temp (TT-110)</display_name>
      <visible>true</visible>
      <name>CWM-CWS02:WtrC-TT-110:MeasValue</name>
      <axis>0</axis>
      <trace_type>LINE</trace_type>
      <linewidth>2</linewidth>
      <color><red>200</red><green>60</green><blue>40</blue></color>
      <period>0.0</period>
      <scan_period>1.0</scan_period>
      <ring_size>5000</ring_size>
      <waveform_index>0</waveform_index>
      <request>RAW</request>
    </pv>
    <pv>
      <display_name>Vessel level (LI-103)</display_name>
      <visible>true</visible>
      <name>CWM-CWS02:WtrC-LI-103:MeasValue</name>
      <axis>1</axis>
      <trace_type>LINE</trace_type>
      <linewidth>2</linewidth>
      <color><red>40</red><green>160</green><blue>60</blue></color>
      <period>0.0</period>
      <scan_period>1.0</scan_period>
      <ring_size>5000</ring_size>
      <waveform_index>0</waveform_index>
      <request>RAW</request>
    </pv>
    <pv>
      <display_name>Pump speed (P-101)</display_name>
      <visible>true</visible>
      <name>CWM-CWS02:WtrC-P-101:Speed</name>
      <axis>2</axis>
      <trace_type>LINE</trace_type>
      <linewidth>2</linewidth>
      <color><red>140</red><green>80</green><blue>180</blue></color>
      <period>0.0</period>
      <scan_period>1.0</scan_period>
      <ring_size>5000</ring_size>
      <waveform_index>0</waveform_index>
      <request>RAW</request>
    </pv>
  </pvlist>
</databrowser>
