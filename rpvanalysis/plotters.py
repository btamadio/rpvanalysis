import random
import string
import ROOT
import array
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT.gROOT.LoadMacro('/global/homes/b/btamadio/atlasstyle/AtlasStyle.C')
ROOT.gROOT.LoadMacro('/global/homes/b/btamadio/atlasstyle/AtlasLabels.C')
ROOT.SetAtlasStyle()
ROOT.gROOT.SetBatch()
def get_random_string(N=10):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def get_region_label(region_str):
    if region_str == '3jCR':
        lines = ['N_{jet} = 3','control region']
        return make_splitline(lines)
    if region_str.startswith('UDR1'):
        region_str = '2jLJG400'+region_str[4:]
    elif region_str.startswith('UDR2'):
        region_str = '4js1LJL400'+region_str[4:]
    lines = []
    if '2j' in region_str:
        lines.append('N_{jet} = 2')
    elif '3j' in region_str:
        lines.append('N_{jet} = 3')
    elif '4j' in region_str:
        lines.append('N_{jet} = 4')
    elif '5j' in region_str:
        lines.append('N_{jet} #geq 5')
    if 's0' in region_str:
        lines.append('N_{soft jet} = 0')
    elif 's1' in region_str:
        lines.append('N_{soft jet} #geq 1')
    if 'VR' in region_str:
        lines.append('|#Delta #eta| > 1.4')
    elif 'SR' in region_str:
        lines.append('|#Delta #eta| < 1.4')
    if 'b0' in region_str:
        lines.append('b-veto')
    elif 'b1' in region_str:
        lines.append('b-tag')
    if 'LJL400' in region_str:
        lines.append('p_{T}^{lead} < 0.4 TeV')
    elif 'LJG400' in region_str:
        lines.append('p_{T}^{lead} > 0.4 TeV')
    if 'bM' in region_str:
        lines.append('b-matched')
    elif 'bU' in region_str:
        lines.append('non-b-matched')
    if 'l1' in region_str:
        lines.append('leading jet')
    elif 'l2' in region_str:
        lines.append('sub-leading jets')
    label = make_splitline(lines)
    return label

def make_splitline(lines):
    if len(lines) == 0:
        return ''
    elif len(lines) == 1:
        return lines[0]
    else:
        return '#splitline{'+make_splitline(lines[:-1])+'}{'+lines[-1]+'}'

def plot_template_compare(temp_1,temp_2,template_type,plot_path,canvas,lumi_label,mc_label,pt_min,pt_max,eta_min,eta_max,temp_bin):
    h1 = ROOT.TH1F('temp_1','temp_1',len(temp_1.bin_centers),array.array('d',temp_1.bin_edges))
    h2 = ROOT.TH1F('temp_2','temp_2',len(temp_2.bin_centers),array.array('d',temp_2.bin_edges))
    for i in range(h1.GetNbinsX()):
        bin = i+1
        h1.SetBinContent(bin,temp_1.sumw[i])
        h2.SetBinContent(bin,temp_2.sumw[i])
        h1.SetBinError(bin,np.sqrt(temp_1.sumw2[i]))
        h2.SetBinError(bin,np.sqrt(temp_2.sumw2[i]))
    h1.Scale(1./h1.Integral())
    h2.Scale(1./h2.Integral())
    canvas.cd()
    rand_str = get_random_string()
    p1 = ROOT.TPad('p1_'+rand_str,'p1_'+rand_str,0,0.3,1,1.0)
    p1.SetBottomMargin(0.01)
    p1.Draw()
    p1.cd()
    p1.SetLogy()

    h1.SetMarkerStyle(20)
    h1.SetMarkerColor(ROOT.kBlack)
    h1.SetLineColor(ROOT.kBlack)

    h2.SetMarkerStyle(20)
    h2.SetLineColor(ROOT.kRed)
    h2.SetMarkerColor(ROOT.kRed)

    h1.GetYaxis().SetTitle('fraction of jets')
    h1.GetYaxis().SetTitleSize(20)
    h1.GetYaxis().SetTitleFont(43)
    h1.GetYaxis().SetTitleOffset(1.55)
    h1.GetYaxis().SetLabelFont(43)
    h1.GetYaxis().SetLabelSize(15)    


    h1.Draw('e ')
    h2.Draw('e same')

    file_name = 'plot_template_'+str(temp_bin)+'.png'
    full_path = plot_path.rstrip('/')+'/templates/'
    ROOT.ATLASLabel(0.25,0.85,'Internal',0.05,0.115,1)

    lat = ROOT.TLatex()
    if mc_label:
        lat.DrawLatexNDC(0.25,0.78,lumi_label+' fb^{-1} '+mc_label)
    else:
        lat.DrawLatexNDC(0.2,0.78,'#sqrt{s} = 13 TeV, '+lumi_label+' fb^{-1}')

    lat.DrawLatexNDC(0.5,0.3,'%i GeV < p_{T} < %i GeV' % ( int(1000*pt_min),int(1000*pt_max)))
    lat.DrawLatexNDC(0.5,0.15,'%.1f < |#eta| < %.1f' % (eta_min,eta_max))

    labels = ('leading 2 jets','3rd leading jet')
    if template_type == 0:
        labels = ('b-match','non-b-match')
    leg = ROOT.TLegend(0.6,0.55,0.8,0.75)

    leg.AddEntry(h1,labels[0],'LP')
    leg.AddEntry(h2,labels[1],'LP')

    leg.SetLineColor(0)
    leg.SetTextSize(0.05)
    leg.SetShadowColor(0)
    leg.SetFillStyle(0)
    leg.SetFillColor(0)
    leg.Draw()
    canvas.cd()

    p2 = ROOT.TPad('p2_'+rand_str,'p2_'+rand_str,0,0.05,1,0.3)
    p2.SetTopMargin(0)
    p2.SetBottomMargin(0.2)
    p2.SetGridy()
    p2.Draw()
    p2.cd()

    ratio_hist = h1.Clone('ratio_hist')
    ratio_hist.Divide(h2)
    for i in range(ratio_hist.GetNbinsX()):
        bin=i+1
        a = temp_1.sumw[i]
        b = temp_2.sumw[i]
        if a==0 or b==0:
            ratio_hist.SetBinContent(bin,0)
        else:
            sigma_a = np.sqrt( temp_1.sumw2[i] )
            sigma_b = np.sqrt( temp_2.sumw2[i] )
            err = np.sqrt( (sigma_a*sigma_a)/(a*a)+ (sigma_b*sigma_b) / (b*b) )
            ratio_hist.SetBinError(bin,err)

    ratio_hist.Draw('e')

    ratio_hist.GetYaxis().SetTitle('ratio')
    ratio_hist.SetMinimum(0.78)
    ratio_hist.SetMaximum(1.22)
    ratio_hist.GetYaxis().SetNdivisions(505)
    ratio_hist.GetYaxis().SetTitleSize(18)
    ratio_hist.GetYaxis().SetTitleFont(43)
    ratio_hist.GetYaxis().SetTitleOffset(1.55)
    ratio_hist.GetYaxis().SetLabelFont(43)
    ratio_hist.GetYaxis().SetLabelSize(15)
    ratio_hist.GetXaxis().SetTitleSize(18)
    ratio_hist.GetXaxis().SetTitleFont(43)
    ratio_hist.GetXaxis().SetTitleOffset(4)
    ratio_hist.GetXaxis().SetLabelFont(43)
    ratio_hist.GetXaxis().SetLabelSize(15)
    ratio_hist.GetXaxis().SetTitle('log(m/p_{T})')
    ratio_hist.SetMinimum(0.0)
    ratio_hist.SetMaximum(1.7)

    print('Saving plot to %s'%full_path+file_name)
    canvas.Print(full_path+file_name)
    os.system('chmod a+r ' + full_path+file_name)

    return(h1,h2)

def plot_MJ(MJ_hists,scale_factor,sr_yields,plot_path,canvas,region_str,MJ_bins,lumi_label,mc_label,blinded,MJ_cut):
    kin_sumw,kin_sumw2,dress_nom_matrix,dress_syst_matrix = MJ_hists

    kin_yield, kin_uncert, pred_yield, pred_stat, pred_syst = sr_yields
    n_systs = dress_syst_matrix.shape[0]
    n_bins = len(MJ_bins)-1

    kin_stat_err = np.sqrt( kin_sumw2 )
    dress_nom_sumw = np.mean( dress_nom_matrix, axis=0)
    dress_stat_err = np.std( dress_nom_matrix, axis = 0)
    dress_syst_sumw = np.zeros( (n_systs,n_bins) )

    for i in range(n_systs):
        dress_syst_sumw[i] = np.mean( dress_syst_matrix[i], axis = 0)
    rand_str = get_random_string()
    canvas.cd()
    p1 = ROOT.TPad('p1_'+rand_str,'p1_'+rand_str,0,0.3,1,1.0)
    p1.SetBottomMargin(0.01)
    p1.Draw()
    p1.cd()
    p1.SetLogy()

    err_hist = ROOT.TH1F('err_hist','err_hist',n_bins,array.array('d',MJ_bins))
    kin_hist = ROOT.TH1F('kin_hist','kin_hist',n_bins,array.array('d',MJ_bins))
    dressed_hist = ROOT.TH1F('dressed_hist','dressed_hist',n_bins,array.array('d',MJ_bins))
    dressed_hist_up = ROOT.TH1F('dressed_hist_up','dressed_hist_up',n_bins,array.array('d',MJ_bins))
    dressed_hist_down = ROOT.TH1F('dressed_hist_down','dressed_hist_down',n_bins,array.array('d',MJ_bins))

    for i in range(n_bins):
        bin = i+1
        if blinded and kin_hist.GetBinLowEdge(bin) >= MJ_cut:
            kin_hist.SetBinContent(bin,0)
            kin_hist.SetBinError( bin,0)
        else:
            kin_hist.SetBinContent(bin,kin_sumw[i])
            kin_hist.SetBinError( bin, kin_stat_err[i] )
        dressed_hist.SetBinContent(bin, dress_nom_sumw[i])
        err_hist.SetBinContent(bin,dress_nom_sumw[i])
        err_syst = 0.0
        err_stat = dress_stat_err[i]
        for j in range(n_systs):
            bin_err = dress_syst_sumw[j][i] - dress_nom_sumw[i]
            err_syst += abs(bin_err)
        tot_err = np.sqrt( err_stat*err_stat + err_syst*err_syst)
        dressed_hist_up.SetBinContent(bin,dressed_hist.GetBinContent(bin) + err_syst)
        dressed_hist_down.SetBinContent(bin,dressed_hist.GetBinContent(bin) - err_syst)
        err_hist.SetBinError(bin,tot_err)
    print('plotting...')
    hist_list = [err_hist,dressed_hist,dressed_hist_up,dressed_hist_down]
    [hist.Scale(scale_factor) for hist in hist_list]

    err_hist.SetMarkerSize(0.001)
    err_hist.SetLineColor(ROOT.kRed)
    err_hist.SetFillColor(ROOT.kRed)
    err_hist.SetFillStyle(3002)

    dressed_hist.SetLineColor(ROOT.kRed)
    dressed_hist.SetLineWidth(2)
    dressed_hist.SetFillStyle(3002)

    kin_hist.SetLineColor(ROOT.kBlack)
    kin_hist.SetLineWidth(2)
    kin_hist.SetMarkerStyle(20)
    kin_hist.SetBinErrorOption(1)

    dressed_hist_up.SetLineColor(ROOT.kGray+1)
    dressed_hist_up.SetLineWidth(2)
    dressed_hist_down.SetLineColor(ROOT.kGray+1)
    dressed_hist_down.SetLineWidth(2)            

    err_hist.SetMinimum(0.5)
    err_hist.SetMaximum( np.ceil(err_hist.GetMaximum()/10000 )*10000)

    err_hist.Draw('e2')
    dressed_hist.Draw('same')
    kin_hist.Draw('same ep')
    dressed_hist_up.Draw('hist same')
    dressed_hist_down.Draw('hist same')

    err_hist.GetYaxis().SetTitle('Events')
    err_hist.GetYaxis().SetTitleSize(20)
    err_hist.GetYaxis().SetTitleFont(43)
    err_hist.GetYaxis().SetTitleOffset(1.55)
    err_hist.GetYaxis().SetLabelFont(43)
    err_hist.GetYaxis().SetLabelSize(15)    

    ROOT.ATLASLabel(0.35,0.85,'Internal',0.05,0.115,1)
    lat = ROOT.TLatex()
    if mc_label:
        lat.DrawLatexNDC(0.35,0.78,lumi_label+' fb^{-1} '+mc_label)
        lat.DrawLatexNDC(0.6,0.82,'#splitline{n_{pred} = %.1f #pm %.1f #pm %.1f}{n_{obs} = %.1f #pm %.1f}' % (pred_yield,pred_stat,pred_syst,kin_yield,kin_uncert))
    else:
        lat.DrawLatexNDC(0.3,0.78,'#sqrt{s} = 13 TeV, '+lumi_label+' fb^{-1}')
        if blinded:
            lat.DrawLatexNDC(0.6,0.82,'n_{pred} = %.1f #pm %.1f #pm %.1f' % (pred_yield,pred_stat,pred_syst))
        else:
            lat.DrawLatexNDC(0.6,0.82,'#splitline{n_{pred} = %.1f #pm %.1f #pm %.1f}{n_{obs} = %i}' % (pred_yield,pred_stat,pred_syst,kin_yield))
    lat.DrawLatexNDC(0.24,0.28,get_region_label(region_str))

    leg = ROOT.TLegend(0.6,0.55,0.8,0.75)
    leg.AddEntry(kin_hist,'Kinematic','LP')
    leg.AddEntry(err_hist,'Prediction #pm 1#sigma','LF')
    leg.AddEntry(dressed_hist_up,'Syst. #pm 1#sigma','L')
    leg.SetLineColor(0)
    leg.SetTextSize(0.05)
    leg.SetShadowColor(0)
    leg.SetFillStyle(0)
    leg.SetFillColor(0)
    leg.Draw()
    

    canvas.cd()

    # #ratio plot
    p2 = ROOT.TPad('p2_'+rand_str,'p2_'+rand_str,0,0.05,1,0.3)
    p2.SetTopMargin(0)
    p2.SetBottomMargin(0.2)
    p2.SetGridy()
    p2.Draw()
    p2.cd()

    ratio_kin_hist = kin_hist.Clone()
    ratio_dressed_hist = err_hist.Clone()
    ratio_dressed_hist_up = dressed_hist_up.Clone()
    ratio_dressed_hist_down = dressed_hist_down.Clone()

    ratio_dressed_hist_up.Divide(dressed_hist)
    ratio_dressed_hist_down.Divide(dressed_hist)
    ratio_dressed_hist.Divide(dressed_hist)

    for bin in range(1,ratio_kin_hist.GetNbinsX()+1):
        if dressed_hist.GetBinContent(bin) > 0:
            ratio_dressed_hist.SetBinError( bin, ratio_dressed_hist.GetBinError(bin) / ratio_dressed_hist.GetBinContent(bin))
        else:
            ratio_dressed_hist.SetBinError(bin,1)
        if err_hist.GetBinContent(bin) > 0:
            ratio_kin_hist.SetBinError(bin,ratio_kin_hist.GetBinError(bin) / err_hist.GetBinContent(bin))
            ratio_kin_hist.SetBinContent(bin,ratio_kin_hist.GetBinContent(bin) / err_hist.GetBinContent(bin))
        else:
            ratio_kin_hist.SetBinContent(bin,0)
            ratio_kin_hist.SetBinError(bin,1)
    ratio_dressed_hist.Draw('e2')
    ratio_kin_hist.Draw('e0 same')
    ratio_dressed_hist_up.Draw('hist same')
    ratio_dressed_hist_down.Draw('hist same')

    ratio_dressed_hist.GetYaxis().SetTitle('Kin/Pred')
    ratio_dressed_hist.SetMinimum(0.78)
    ratio_dressed_hist.SetMaximum(1.22)
    ratio_dressed_hist.GetYaxis().SetNdivisions(505)
    ratio_dressed_hist.GetYaxis().SetTitleSize(18)
    ratio_dressed_hist.GetYaxis().SetTitleFont(43)
    ratio_dressed_hist.GetYaxis().SetTitleOffset(1.55)
    ratio_dressed_hist.GetYaxis().SetLabelFont(43)
    ratio_dressed_hist.GetYaxis().SetLabelSize(15)
    ratio_dressed_hist.GetXaxis().SetTitleSize(18)
    ratio_dressed_hist.GetXaxis().SetTitleFont(43)
    ratio_dressed_hist.GetXaxis().SetTitleOffset(4)
    ratio_dressed_hist.GetXaxis().SetLabelFont(43)
    ratio_dressed_hist.GetXaxis().SetLabelSize(15)
    ratio_dressed_hist.GetXaxis().SetTitle('M_{J}^{#Sigma} [TeV]')
    ratio_dressed_hist.SetMinimum(0.0)
    ratio_dressed_hist.SetMaximum(1.7)

    return_list = [kin_hist,
                   dressed_hist,
                   err_hist,
                   dressed_hist_up,
                   dressed_hist_down,
                   ratio_kin_hist,
                   ratio_dressed_hist,
                   ratio_dressed_hist_up,
                   ratio_dressed_hist_down,
                   lat,leg]

    canvas.cd()
    canvas.Update()
    file_name = 'plot_MJ.png'
    full_path = plot_path.rstrip('/')+'/'+region_str+'/'
    print('Saving plot to %s'%full_path+file_name)
    canvas.Print(full_path+file_name)
    os.system('chmod a+r ' + full_path+file_name)
    return return_list 

def plot_response(response,plot_path,canvas,region_str,pt_bins,eta_bins,lumi_label='36.5',mc_label=''):
    dressed_mean,kin_mean,err = response

    dressed_mean = np.nan_to_num(dressed_mean)
    kin_mean = np.nan_to_num(kin_mean)
    err = np.nan_to_num(err)

    rand_str = get_random_string()
    canvas.cd()      
    p1 = ROOT.TPad('p1_'+rand_str,'p1_'+rand_str,0,0.3,1,1.0)
    p1.SetBottomMargin(0.01)
    p1.Draw()
    p1.cd()

    n_bins = len(pt_bins)-1
    kin_hist = ROOT.TH1F('kin_hist','kin_hist',n_bins,array.array('d',pt_bins))
    dressed_hist = ROOT.TH1F('dressed_hist','dressed_hist',n_bins,array.array('d',pt_bins))
    kin_hist.SetDirectory(0)
    dressed_hist.SetDirectory(0)
    for i in range(n_bins):
        bin = i+1
        kin_hist.SetBinContent(bin,kin_mean[i])
        kin_hist.SetBinError(bin,err[i])
        dressed_hist.SetBinContent(bin,dressed_mean[i])
    print('plotting...')

    dressed_hist.SetLineColor(ROOT.kRed)
    dressed_hist.SetLineWidth(2)
    dressed_hist.SetFillStyle(3002)
    dressed_hist.SetMinimum(0.0)
    dressed_hist.SetMaximum(0.25)

    kin_hist.SetLineColor(ROOT.kBlack)
    kin_hist.SetLineWidth(2)
    kin_hist.SetMarkerStyle(20)
    kin_hist.SetMarkerSize(0.01)

    dressed_hist.Draw()
    kin_hist.Draw('same ep')
    dressed_hist.GetYaxis().SetTitle('<m_{jet}> [TeV]')
    dressed_hist.GetYaxis().SetTitleSize(20)
    dressed_hist.GetYaxis().SetTitleFont(43)
    dressed_hist.GetYaxis().SetTitleOffset(1.55)
    dressed_hist.GetYaxis().SetLabelFont(43)
    dressed_hist.GetYaxis().SetLabelSize(15)

    #various labels
    ROOT.ATLASLabel(0.25,0.85,'Internal',0.05,0.115,1)
    lat = ROOT.TLatex()
    if mc_label:
        lat.DrawLatexNDC(0.25,0.78,lumi_label+' fb^{-1} '+mc_label)
    else:
        lat.DrawLatexNDC(0.25,0.78,'#sqrt{s} = 13 TeV, '+lumi_label+' fb^{-1}')
    lat.DrawLatexNDC(0.24,0.42,get_region_label(region_str))
    
    if 'e1' in region_str:
        lat.DrawLatexNDC(0.7,0.18,'|#eta| < %.1f'%eta_bins[1])
    elif 'e2' in region_str:
        lat.DrawLatexNDC(0.7,0.18,'%.1f < |#eta| < %.1f' % (eta_bins[1],eta_bins[2]))
    elif 'e3' in region_str:
        lat.DrawLatexNDC(0.7,0.18,'%.1f < |#eta| < %.1f' % (eta_bins[2],eta_bins[3]))
    elif 'e4' in region_str:
        lat.DrawLatexNDC(0.7,0.18,'%.1f < |#eta| < %.1f' % (eta_bins[3],eta_bins[4]))

    leg = ROOT.TLegend(0.65,0.7,0.85,0.9)
    leg.AddEntry(kin_hist,'Kinematic','LP')
    leg.AddEntry(dressed_hist,'Prediction #pm 1#sigma','LF')
    leg.SetLineColor(0)
    leg.SetTextSize(0.05)
    leg.SetShadowColor(0)
    leg.SetFillStyle(0)
    leg.SetFillColor(0)
    leg.Draw()
    canvas.cd()

    #ratio plot
    p2 = ROOT.TPad('p2_'+rand_str,'p2_'+rand_str,0,0.05,1,0.3)
    p2.SetTopMargin(0)
    p2.SetBottomMargin(0.2)
    p2.SetGridy()
    p2.Draw()
    p2.cd()

    ratio_hist = kin_hist.Clone('ratio_hist')

    for bin in range(1,ratio_hist.GetNbinsX()+1):
        if dressed_hist.GetBinContent(bin) > 0:
            ratio_hist.SetBinError(bin,ratio_hist.GetBinError(bin) / dressed_hist.GetBinContent(bin))
            ratio_hist.SetBinContent(bin,ratio_hist.GetBinContent(bin) / dressed_hist.GetBinContent(bin))
        else:
            ratio_hist.SetBinError(bin,0)
            ratio_hist.SetBinContent(bin,0)
    ratio_hist.Draw('e0')

    ratio_hist.GetYaxis().SetTitle('Kin/Pred')
    ratio_hist.SetMinimum(0.78)
    ratio_hist.SetMaximum(1.22)
    ratio_hist.GetYaxis().SetNdivisions(505)
    ratio_hist.GetYaxis().SetTitleSize(18)
    ratio_hist.GetYaxis().SetTitleFont(43)
    ratio_hist.GetYaxis().SetTitleOffset(1.55)
    ratio_hist.GetYaxis().SetLabelFont(43)
    ratio_hist.GetYaxis().SetLabelSize(15)

    ratio_hist.GetXaxis().SetTitleSize(18)
    ratio_hist.GetXaxis().SetTitleFont(43)
    ratio_hist.GetXaxis().SetTitleOffset(4)
    ratio_hist.GetXaxis().SetLabelFont(43)
    ratio_hist.GetXaxis().SetLabelSize(15)
    ratio_hist.GetXaxis().SetTitle('jet p_{T} [TeV]')
    canvas.cd()
    canvas.Update()

    return_list = [dressed_hist,kin_hist,ratio_hist,lat,leg]

    file_name = 'plot_mass_response.png'
    full_path = plot_path+'/'+region_str+'/'+file_name
    print('Saving plot to %s'%full_path)
    canvas.Print(full_path)
    os.system('chmod a+r %s/%s/*' % (plot_path,region_str))
    return [dressed_hist,kin_hist,ratio_hist,lat,leg]

def plot_hist(h):
    plt.figure()
    plt.bar(h[1][:-1],h[0],width=h[1][1]-h[1][0])
    plt.show()

def plot_template(t):
    h = (t.sumw_neg,t.bin_edges)
    plt.bar(h[1][:-1],h[0],width=h[1][1]-h[1][0])
    plt.xlabel('$log(m/p_T)$')
    plt.ylabel('jets')
    plt.show()

def make_webpage(plot_path):
    url = plot_path.rstrip('/')+'/plots.html'
    with open(url,'w') as f:
        f.write('<HTML><BODY> <CENTER><TABLE border=3>')

        f.write('<TR>')
        for region_str in ['3jb0','3jVRb1bU','3jVRb1bM','3jCR']:
            f.write('<TD><img src = "%s/plot_mass_response.png" height = "800" width = "800"></TD>' % region_str)
        f.write('</TR>')


        f.write('<TR>')
        for region_str in ['UDR1e1','UDR1e2','UDR1e3','UDR1e4']:
            f.write('<TD><img src = "%s/plot_mass_response.png" height = "800" width = "800"></TD>' % region_str)
        f.write('</TR>')

        f.write('<TR>')
        for region_str in ['UDR2e1','UDR2e2','UDR2e3','UDR2e4']:
            f.write('<TD><img src = "%s/plot_mass_response.png" height = "800" width = "800"></TD>' % region_str)
        f.write('</TR>')

        f.write('<TR>')
        for region_str in ['UDR1bU','UDR1bM','UDR2bU','UDR2bM']:
            f.write('<TD><img src = "%s/plot_mass_response.png" height = "800" width = "800"></TD>' % region_str)
        f.write('</TR>')

        f.write('<TR>')
        for region_str in ['4jVRb1','4jVRb9','4jSRb1','4jSRb9']:
            f.write('<TD><img src = "%s/plot_MJ.png" height = "800" width = "800"></TD>' % region_str)
        f.write('</TR>')

        f.write('<TR>')
        for region_str in ['5jVRb1','5jVRb9','5jSRb1','5jSRb9']:
            f.write('<TD><img src = "%s/plot_MJ.png" height = "800" width = "800"></TD>' % region_str)
        f.write('</TR>')

        f.write('</TABLE></HTML>')
        
