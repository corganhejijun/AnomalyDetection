
moduleList = ["bhatt_hist", "chi_hist", "corr_hist", "fid", "hell_hist", "inter_hist", "psnr", "ssim"];

for v = moduleList
    showROC(v);
end

function showROC(module)
    filename = "csv/" + module + "_ROC_curve.csv";
    table = readtable(filename);
    a = table2array(table);
    y = a(:,1);
    x = a(:,2);

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
