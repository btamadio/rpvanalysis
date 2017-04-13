import random
import string
import ROOT
import array

def get_random_string(N=10):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def plot_response(response,region_string,pt_bins,eta_bin=-1):
    dressed_mean,kin_mean,err = response
    ROOT.gROOT.LoadMacro('/global/homes/b/btamadio/atlasstyle/AtlasStyle.C')
    ROOT.gROOT.LoadMacro('/global/homes/b/btamadio/atlasstyle/AtlasLabels.C')
    ROOT.SetAtlasStyle()
    rand_str = get_random_string()
    can_name = 'c_'+rand_str
    c = ROOT.TCanvas(can_name,can_name,800,800)
    c.cd()      
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
    dressed_hist.Draw()
    dressed_hist.SetMinimum(0.0)
    dressed_hist.SetMaximum(0.25)
    kin_hist.Draw('same ep')

    dressed_hist.SetLineColor(ROOT.kRed)
    dressed_hist.SetLineWidth(2)
    dressed_hist.SetFillStyle(3002)

    kin_hist.SetLineColor(ROOT.kBlack)
    kin_hist.SetLineWidth(2)
    kin_hist.SetMarkerStyle(20)
    kin_hist.SetMarkerSize(0.01)

    ROOT.ATLASLabel(0.35,0.85,'Internal',0.05,0.115,1)
    leg = ROOT.TLegend(0.65,0.7,0.85,0.9)
    leg.AddEntry(kin_hist,'Kinematic','LP')
    leg.AddEntry(dressed_hist,'Prediction #pm 1#sigma','LF')
    leg.SetLineColor(0)
    leg.SetTextSize(0.05)
    leg.SetShadowColor(0)
    leg.SetFillStyle(0)
    leg.SetFillColor(0)
    leg.Draw()
    c.cd()
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
    ratio_hist.SetMinimum(0.8)
    ratio_hist.SetMaximum(1.2)
    ratio_hist.GetYaxis().SetNdivisions(505)
    ratio_hist.GetYaxis().SetTitleSize(20)
    ratio_hist.GetYaxis().SetTitleFont(43)
    ratio_hist.GetYaxis().SetTitleOffset(1.55)
    ratio_hist.GetYaxis().SetLabelFont(43)
    ratio_hist.GetYaxis().SetLabelSize(15)
    ratio_hist.GetXaxis().SetTitleSize(17)
    ratio_hist.GetXaxis().SetTitleFont(43)
    ratio_hist.GetXaxis().SetTitleOffset(3.8)
    ratio_hist.GetXaxis().SetLabelFont(43)
    ratio_hist.GetXaxis().SetLabelSize(15)
    c.cd()
    c.Update()
