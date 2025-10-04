@echo off
REM Quick Plot Generation - Windows Batch Script
echo ========================================
echo   Quick Plot Generation
echo ========================================
echo.

REM Check if data exists
if not exist "evaluation_data\evaluation_results.csv" (
    echo [ERROR] No data found!
    echo Please run evaluations first using: python app.py
    pause
    exit /b 1
)

echo [INFO] Found data file
echo [INFO] Generating plots...
echo.

REM Generate plots
python -c "from visualizer import HypothesisVisualizer; viz = HypothesisVisualizer('evaluation_data/evaluation_results.csv'); viz.generate_all_plots()"

if %ERRORLEVEL% == 0 (
    echo.
    echo ========================================
    echo   SUCCESS! Plots Generated
    echo ========================================
    echo.
    echo Plots saved to: evaluation_data\visualizations\
    echo.
    echo Files created:
    echo   - metric_comparison.png
    echo   - hypothesis_validation.png  
    echo   - metric_heatmap.png
    echo   - score_distribution.png
    echo.
    echo Opening folder...
    start evaluation_data\visualizations
) else (
    echo.
    echo [ERROR] Plot generation failed!
    echo Please check if matplotlib and seaborn are installed:
    echo   pip install matplotlib seaborn
)

echo.
pause
