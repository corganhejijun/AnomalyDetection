unused = [];
moduleList = ["fid", "psnr", "ssim", "d_real_1", "d_both_1", "d_fake_1"];

for v = moduleList
    showROC(v);
end

error = readtable('cal_error.csv');
a = table2array(error);
d_both_1_error = a(:,12);
d_fake_1_error = a(:, 9);
d_real_1_error = a(:, 6);
value = std(d_both_1_error);
normal = getValue(value);
disp(['d_both_1 std=', num2str(value), ' ', normal]);
value = std(d_fake_1_error);
normal = getValue(value);
disp(['d_fake_1 std=', num2str(value), ' ', normal]);
value = std(d_real_1_error);
normal = getValue(value);
disp(['d_real_1 std=', num2str(value), ' ', normal]);

function showROC(module)
    filename = "test_csv/" + module + "_ROC_curve.csv";
    table = readtable(filename);
    a = table2array(table);
    x = a(:,1);
    y = a(:,2);

    figure1 = figure;
    axes1 = axes('Parent',figure1);
    hold(axes1,'on');

    plot(x, y);

    auc = trapz(x, y);

    title(module + " auc = " + num2str(auc) ,'Interpreter','none');

    xlim(axes1,[0 1]);
    ylim(axes1,[0 1]);

    box(axes1,'on');
end

function normal = getValue(value)
    normal = 'normal';
    if (value < 0)
        disp('d_both_1 std value < 0 error');
    end
    if (value > 0.0001)
        normal = 'abnormal';
    end
end
