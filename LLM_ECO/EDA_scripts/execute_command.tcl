# execute command in iteration i
restore_session eco_session_${i}
# Fill the command here
{}
# After executing command, get report
report_qor > reports/report_qor_${i+1}.txt
report_power > reports/report_power_${i+1}.txt
save_session eco_session_${i+1}
exit