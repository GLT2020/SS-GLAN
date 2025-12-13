pragma solidity ^0.5.7;
interface tokenRecipient { 
    function receiveApproval(address _from, uint256 _value, address _token, bytes calldata _extraData) external; 
}
contract  MD{
  function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
  string public name;
  function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
  string public symbol;
  function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
  uint8 public decimals = 18;
  function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public totalSupply;
  function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => uint256) public balanceOf;
  function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => mapping (address => uint256)) public allowance;
  function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed from, address indexed to, uint256 value);
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event Approval(address indexed _owner, address indexed _spender, uint256 _value);
  function bug_unchk_send13() payable public{
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
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function _transfer(address _from, address _to, uint _value) internal {
        require(_to != address(0x0));
        require(balanceOf[_from] >= _value);
        require(balanceOf[_to] + _value >= balanceOf[_to]);
        uint previousBalances = balanceOf[_from] + balanceOf[_to];
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(_from, _to, _value);
        assert(balanceOf[_from] + balanceOf[_to] == previousBalances);
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint256 _value) public returns (bool success) {
        _transfer(msg.sender, _to, _value);
        return true;
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(_value <= allowance[_from][msg.sender]);     
        allowance[_from][msg.sender] -= _value;
        _transfer(_from, _to, _value);
        return true;
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address _spender, uint256 _value) public
        returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function approveAndCall(address _spender, uint256 _value, bytes memory _extraData)
        public
        returns (bool success) {
        tokenRecipient spender = tokenRecipient(_spender);
        if (approve(_spender, _value)) {
            spender.receiveApproval(msg.sender, _value, address(this), _extraData);
            return true;
        }
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function burn(uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value);   
        balanceOf[msg.sender] -= _value;            
        totalSupply -= _value;                      
        emit Burn(msg.sender, _value);
        return true;
    }
function bug_unchk_send30() payable public{
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
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
}