def decodeHiddenMessage(fileName):
  import DISCOVERY
  df = DISCOVERY.df_image(fileName)
  df = df[df["y"] <= 26]
  df["mod2"] = df["b"] % 2
  df = df[df["mod2"] == 0]
  df["charCode"] = df["y"] + 65
  df["character"] = df["charCode"].apply(chr)
  message = df.character.str.cat()
  message = message.replace("[", " ")
  return message
