pragma solidity >=0.4.22 <0.6.0;
contract Ownable {
function bug_unchk_send29() payable public{
      msg.sender.transfer(1 ether);}
  address public owner;
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
  constructor () public {
    owner = msg.sender;
  }
function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
}
contract TokenERC20 {
  function bug_unchk_send6() payable public{
      msg.sender.transfer(1 ether);}
  string public name;
  function bug_unchk_send16() payable public{
      msg.sender.transfer(1 ether);}
  string public symbol;
  function bug_unchk_send24() payable public{
      msg.sender.transfer(1 ether);}
  uint8 public decimals = 18;
  function bug_unchk_send5() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public totalSupply;
  function bug_unchk_send15() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => uint256) public balanceOf;
  function bug_unchk_send28() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => mapping (address => uint256)) public allowance;
  function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed from, address indexed to, uint256 value);
  function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event Approval(address indexed _owner, address indexed _spender, uint256 _value);
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event Burn(address indexed from, uint256 value);
    constructor(
        uint256 initialSupply,
        string memory tokenName,
        string memory tokenSymbol
    ) public {
        totalSupply = initialSupply * 10 ** uint256(decimals);  
        balanceOf[msg.sender] = totalSupply;                    
        name = tokenName;                                       
        symbol = tokenSymbol;                                   
    }
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
    function _transfer(address _from, address _to, uint _value) internal {
        require(_to != address(0x0));
        require(balanceOf[_from] >= _value);
        require(balanceOf[_to] + _value > balanceOf[_to]);
        uint previousBalances = balanceOf[_from] + balanceOf[_to];
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(_from, _to, _value);
        assert(balanceOf[_from] + balanceOf[_to] == previousBalances);
    }
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint256 _value) public returns (bool success) {
        _transfer(msg.sender, _to, _value);
        return true;
    }
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(_value <= allowance[_from][msg.sender]);     
        allowance[_from][msg.sender] -= _value;
        _transfer(_from, _to, _value);
        return true;
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address _spender, uint256 _value) public
        returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
    function burn(uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value);   
        balanceOf[msg.sender] -= _value;            
        totalSupply -= _value;                      
        emit Burn(msg.sender, _value);
        return true;
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function burnFrom(address _from, uint256 _value) public returns (bool success) {
        require(balanceOf[_from] >= _value);                
        require(_value <= allowance[_from][msg.sender]);    
        balanceOf[_from] -= _value;                         
        allowance[_from][msg.sender] -= _value;             
        totalSupply -= _value;                              
        emit Burn(_from, _value);
        return true;
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
}
contract YFT is Ownable, TokenERC20 {
  function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public sellPrice;
  function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public buyPrice;
  function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => bool) public frozenAccount;
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event FrozenFunds(address target, bool frozen);
    constructor(
        uint256 initialSupply,
        string memory tokenName,
        string memory tokenSymbol
    ) TokenERC20(initialSupply, tokenName, tokenSymbol) public {}
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function _transfer(address _from, address _to, uint _value) internal {
        require (_to != address(0x0));                          
        require (balanceOf[_from] >= _value);                   
        require (balanceOf[_to] + _value >= balanceOf[_to]);    
        require(!frozenAccount[_from]);                         
        require(!frozenAccount[_to]);                           
        balanceOf[_from] -= _value;                             
        balanceOf[_to] += _value;                               
        emit Transfer(_from, _to, _value);
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function mintToken(address target, uint256 mintedAmount) onlyOwner public {
        balanceOf[target] += mintedAmount;
        totalSupply += mintedAmount;
        emit Transfer(address(0), address(this), mintedAmount);
        emit Transfer(address(this), target, mintedAmount);
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function freezeAccount(address target, bool freeze) onlyOwner public {
        frozenAccount[target] = freeze;
        emit FrozenFunds(target, freeze);
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function setPrices(uint256 newSellPrice, uint256 newBuyPrice) onlyOwner public {
        sellPrice = newSellPrice;
        buyPrice = newBuyPrice;
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function buy() payable public {
        uint amount = msg.value / buyPrice;                 
        _transfer(address(this), msg.sender, amount);       
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function sell(uint256 amount) public {
        address myAddress = address(this);
        require(myAddress.balance >= amount * sellPrice);   
        _transfer(msg.sender, address(this), amount);       
        msg.sender.transfer(amount * sellPrice);            
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
}