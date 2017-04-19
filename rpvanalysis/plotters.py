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
    lines = []
    if '3j' in region_str:
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
    if 'bM' in region_str:
        lines.append('b-matched')
    elif 'bU' in region_str:
        lines.append('non-b-matched')
    label = ''
    if len(lines) == 1:
        label = lines[0]
    elif len(lines) == 2:
        label = '#splitline{'+lines[0]+'}{'+lines[1]+'}'
    elif len(lines) == 3:
        label = '#splitline{#splitline{'+lines[0]+'}{'+lines[1]+'}}{'+lines[2]+'}'
    elif len(lines) == 4:
        label = '#splitline{#splitline{'+lines[0]+'}{'+lines[1]+'}}{#splitline{'+lines[2]+'}{'+lines[3]+'}}'
    elif len(lines) == 5:
        label = '#splitline{#splitline{#splitline{'+lines[0]+'}{'+lines[1]+'}}{#splitline{'+lines[2]+'}{'+lines[3]+'}}}{'+lines[4]+'}'
    return label

def plot_MJ(MJ_hists,plot_path,canvas,region_str,MJ_bins,lumi_label,mc_label):
    kin_sumw,kin_sumw2,dressed_nom_sumw,dressed_shift_sumw = MJ_hists
    rand_str = get_random_string()
    canvas.cd()
    p1 = ROOT.TPad('p1_'+rand_str,'p1_'+rand_str,0,0.3,1,1.0)
    p1.SetBottomMargin(0.01)
    p1.Draw()
    p1.cd()
    p1.SetLogy()
    n_bins = len(MJ_bins)-1

    err_hist = ROOT.TH1F('err_hist','err_hist',n_bins,array.array('d',MJ_bins))
    kin_hist = ROOT.TH1F('kin_hist','kin_hist',n_bins,array.array('d',MJ_bins))
    dressed_hist = ROOT.TH1F('dressed_hist','dressed_hist',n_bins,array.array('d',MJ_bins))

    err_hist.SetDirectory(0)
    kin_hist.SetDirectory(0)
    dressed_hist.SetDirectory(0)

    for i in range(n_bins):
        bin = i+1
        kin_hist.SetBinContent(bin,kin_sumw[i])
        kin_hist.SetBinError( bin,np.sqrt( kin_sumw[i]*kin_sumw[i] / kin_sumw2[i] ) )
    dressed_hist.SetLineColor(ROOT.kRed)
    dressed_hist.SetLineWidth(2)
    dressed_hist.SetFillStyle(3002)
    dressed_hist.SetMinimum(0.0)
    dressed_hist.SetMaximum(0.25)

    kin_hist.SetLineColor(ROOT.kBlack)
    kin_hist.SetLineWidth(2)
    kin_hist.SetMarkerStyle(20)
    kin_hist.SetMarkerSize(0.01)
    kin_hist.Draw('ep')

    #dressed_hist.Draw()
    #kin_hist.Draw('same ep')

    # dressed_hist.GetYaxis().SetTitle('<m_{jet}> [TeV]')

    # dressed_hist.GetYaxis().SetTitleSize(20)
    # dressed_hist.GetYaxis().SetTitleFont(43)
    # dressed_hist.GetYaxis().SetTitleOffset(1.55)
    # dressed_hist.GetYaxis().SetLabelFont(43)
    # dressed_hist.GetYaxis().SetLabelSize(15)    

    ROOT.ATLASLabel(0.35,0.85,'Internal',0.05,0.115,1)
    lat = ROOT.TLatex()
    if mc_label:
        lat.DrawLatexNDC(0.25,0.78,lumi_label+' fb^{-1} '+mc_label)
    else:
        lat.DrawLatexNDC(0.25,0.78,'#sqrt{s} = 13 TeV, '+lumi_label+' fb^{-1}')
    lat.DrawLatexNDC(0.24,0.42,get_region_label(region_str))

    #legend
    # leg = ROOT.TLegend(0.65,0.7,0.85,0.9)
    # leg.AddEntry(kin_hist,'Kinematic','LP')
    # leg.AddEntry(dressed_hist,'Prediction #pm 1#sigma','LF')
    # leg.SetLineColor(0)
    # leg.SetTextSize(0.05)
    # leg.SetShadowColor(0)
    # leg.SetFillStyle(0)
    # leg.SetFillColor(0)
    # leg.Draw()
    # canvas.cd()

    # #ratio plot
    # p2 = ROOT.TPad('p2_'+rand_str,'p2_'+rand_str,0,0.05,1,0.3)
    # p2.SetTopMargin(0)
    # p2.SetBottomMargin(0.2)
    # p2.SetGridy()
    # p2.Draw()
    # p2.cd()

    # ratio_hist = kin_hist.Clone()

    # for bin in range(1,ratio_hist.GetNbinsX()+1):
    #     if dressed_hist.GetBinContent(bin) > 0:
    #         ratio_hist.SetBinError(bin,ratio_hist.GetBinError(bin) / dressed_hist.GetBinContent(bin))
    #         ratio_hist.SetBinContent(bin,ratio_hist.GetBinContent(bin) / dressed_hist.GetBinContent(bin))
    #     else:
    #         ratio_hist.SetBinError(bin,1)
    #         ratio_hist.SetBinContent(bin,0)
    # ratio_hist.Draw('e0')

    # ratio_hist.GetYaxis().SetTitle('Kin/Pred')
    # ratio_hist.SetMinimum(0.78)
    # ratio_hist.SetMaximum(1.22)
    # ratio_hist.GetYaxis().SetNdivisions(505)
    # ratio_hist.GetYaxis().SetTitleSize(18)
    # ratio_hist.GetYaxis().SetTitleFont(43)
    # ratio_hist.GetYaxis().SetTitleOffset(1.55)
    # ratio_hist.GetYaxis().SetLabelFont(43)
    # ratio_hist.GetYaxis().SetLabelSize(15)

    # ratio_hist.GetXaxis().SetTitleSize(18)
    # ratio_hist.GetXaxis().SetTitleFont(43)
    # ratio_hist.GetXaxis().SetTitleOffset(4)
    # ratio_hist.GetXaxis().SetLabelFont(43)
    # ratio_hist.GetXaxis().SetLabelSize(15)
    # ratio_hist.GetXaxis().SetTitle('jet p_{T} [TeV]')
    canvas.Update()
    file_name = 'plot_MJ.png'
    full_path = plot_path+'/'+region_str+'/'+file_name
    print('Saving plot to %s'%full_path)
    canvas.Print(full_path)
    os.system('chmod a+r %s/%s/*' % (plot_path,region_str))
    
def plot_response(response,plot_path,canvas,region_str,pt_bins,lumi_label='36.5',mc_label='',eta_bin=-1):
    dressed_mean,kin_mean,err = response
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
    if eta_bin == 0:
        lat.DrawLatexNDC(0.7,0.18,'|#eta| < 0.5')
    elif eta_bin == 1:
        lat.DrawLatexNDC(0.7,0.18,'0.5 < |#eta| < 1.0')
    elif eta_bin == 2:
        lat.DrawLatexNDC(0.7,0.18,'1.0 < |#eta| < 1.5')
    elif eta_bin == 3:
        lat.DrawLatexNDC(0.7,0.18,'1.5 < |#eta| < 2.0')
    #legend
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

    ratio_hist = kin_hist.Clone()

    for bin in range(1,ratio_hist.GetNbinsX()+1):
        if dressed_hist.GetBinContent(bin) > 0:
            ratio_hist.SetBinError(bin,ratio_hist.GetBinError(bin) / dressed_hist.GetBinContent(bin))
            ratio_hist.SetBinContent(bin,ratio_hist.GetBinContent(bin) / dressed_hist.GetBinContent(bin))
        else:
            ratio_hist.SetBinError(bin,1)
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
    canvas.Update()
    file_name = 'plot_mass_response.png'
    full_path = plot_path+'/'+region_str+'/'+file_name
    print('Saving plot to %s'%full_path)
    canvas.Print(full_path)
    os.system('chmod a+r %s/%s/*' % (plot_path,region_str))
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
    
